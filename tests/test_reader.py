"""Reference loader and closed loop: the stored tables determine the stored voltages."""

from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.reader import DatasetReader, VariantTables
from gridfm_datakit_distr.io.writer import STATE_TABLES, DatasetWriter, part_key, read_states
from gridfm_datakit_distr.model.ids import split_scenario_id
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.stats.physics import physics
from gridfm_datakit_distr.validate.closed_loop import closed_loop
from gridfm_datakit_distr.views.stored import VariantStore
from tests.test_executor import catalog, config


@pytest.fixture(scope="module", params=["full", "mixed"])
def dataset(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp(request.param)
    generate(config(float_precision=request.param), root / "out", catalog=catalog(root), root=root)
    return root / "out"


def test_the_source_tables(dataset: Path) -> None:
    source = pq.read_table(dataset / "variants" / "source")
    schema.check_schema(source, schema.SOURCE, "variants/source")
    y = pq.read_table(dataset / "variants" / "source_yprim")
    schema.check_schema(y, schema.SOURCE_YPRIM, "variants/source_yprim")
    variants = set(pq.read_table(dataset / "variants" / "variants")["variant_id"].to_pylist())
    assert set(source["variant_id"].to_pylist()) == variants
    for vid in variants:  # three conductors at 1 pu of the source's base voltage
        rows = [r for r in source.to_pylist() if r["variant_id"] == vid]
        assert [r["position"] for r in rows] == [0, 1, 2]
        e = np.array([complex(r["e_re"], r["e_im"]) for r in rows])
        assert np.allclose(e[1:] / e[:-1], np.exp(-2j * np.pi / 3))


def test_the_closed_loop_reproduces_the_stored_voltages(dataset: Path) -> None:
    reader = DatasetReader(dataset)
    meta = reader.meta.to_pylist()
    ids = [r["scenario_id"] for r in meta if r["pf_status"] == "converged"]
    assert len(ids) >= len(meta) // 2
    for sc in reader.scenarios(ids):
        result = closed_loop(sc)
        assert result.max_dv_pu <= 1e-6, (sc.variant.variant_id, result)


def test_the_reader_imports_no_engine() -> None:
    code = (
        "import sys, gridfm_datakit_distr.io.reader, gridfm_datakit_distr.validate.closed_loop\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('opendssdirect', 'dss', 'casadi')"
        " or m.startswith('gridfm_datakit_distr.engine')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _read_back(root: Path, cfg: GenerationConfig) -> dict[str, Any]:
    """What the reader, the statistics and the stored view make of a dataset."""
    reader = DatasetReader(root)
    ids = reader.meta.column("scenario_id").to_pylist()
    y = {
        sc.scenario_id: sc.y_state.toarray()
        for sc in reader.scenarios(ids)
        if sc.y_state is not None
    }
    stats = {
        nid: (s.summary(), {k: h.counts.tolist() for k, h in s.histograms.items()})
        for nid, s in physics(root, 0.95, 1.05).items()
    }
    currents = {}
    for i, k in itertools.product((0, 1), (0, 1)):
        for _, scenarios in VariantStore(root, i, k, cfg).chunks():
            currents.update({o.scenario_id: o.i_pd for o in scenarios})
    return {"y": y, "stats": stats, "currents": currents}


def test_an_unfinished_dataset_reads_as_the_finished_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before its state parts are merged a state can sit in several parts (every chunk that met
    it): read as merged, by the same rule. Two layouts: parts only (a run killed before
    finalisation) and a merged file plus newer parts (a resumed run killed again)."""
    cfg, cat = config(), catalog(tmp_path)
    parts = tmp_path / "parts"
    with monkeypatch.context() as m:
        m.setattr(DatasetWriter, "consolidate_states", lambda self: None)
        generate(cfg, parts, catalog=cat, root=tmp_path)
    for t in STATE_TABLES:  # a later chunk of t/reg that met the states of its first chunk
        for p in (parts / t).glob("part-00001-0000-00000.parquet"):
            shutil.copy(p, p.with_name("part-00001-0000-00009.parquet"))
    holders = Counter(
        s
        for p in (parts / "states" / "ybus").glob("part-*.parquet")
        for s in set(pq.read_table(p, columns=["state_id"]).column("state_id").to_pylist())
    )
    assert max(holders.values()) > 1
    finished = tmp_path / "finished"
    shutil.copytree(parts, finished)
    DatasetWriter(finished).consolidate_states()
    mixed, held = tmp_path / "mixed", tmp_path / "held"
    shutil.copytree(parts, mixed)
    later = [
        p
        for t in STATE_TABLES
        for p in (mixed / t).glob("part-*.parquet")
        if p.stem[-5:] != "00000"
    ]
    assert later
    for p in later:
        (held / p.relative_to(mixed)).parent.mkdir(parents=True, exist_ok=True)
        p.rename(held / p.relative_to(mixed))
    DatasetWriter(mixed).consolidate_states()  # chunk 0's states only
    for p in later:
        p.parent.mkdir(parents=True, exist_ok=True)
        (held / p.relative_to(mixed)).rename(p)
    expected = _read_back(finished, cfg)
    assert expected["y"]
    assert expected["currents"]
    for root in (parts, mixed):
        got = _read_back(root, cfg)
        assert got["stats"] == expected["stats"], root.name
        for name in ("y", "currents"):
            assert got[name].keys() == expected[name].keys(), (root.name, name)
            for sid, value in expected[name].items():
                assert np.array_equal(got[name][sid], value), (root.name, name, sid)


def test_a_state_stores_only_its_changes(dataset: Path) -> None:
    """Y_state = the variant's matrix with the state's stored entries in place equals the matrix
    assembled from the state's stored Yprims; a state of the network without controls (t/feeder)
    equals its variant's matrix and stores no entry."""
    reader = DatasetReader(dataset)
    yprim_state = read_states(dataset, "states/yprim_state")
    changed = set(pq.read_table(dataset / "states" / "ybus.parquet")["state_id"].to_pylist())
    seen = set()
    for sc in reader.scenarios(reader.meta.column("scenario_id").to_pylist()):
        if sc.state_id is None or sc.state_id in seen:
            continue
        seen.add(sc.state_id)
        i, k, _ = split_scenario_id(sc.scenario_id)
        stored = VariantTables(dataset, part_key(i, k), sc.variant.variant_id, yprim_state)
        assert sc.y_state is not None
        assert (sc.y_state != stored.model(sc.state_id).ybus()).nnz == 0, sc.state_id
        if sc.variant.variant_id.startswith("t/feeder"):
            assert sc.state_id not in changed
            assert (sc.y_state != sc.variant.y_base).nnz == 0
    assert changed  # t/reg's settled taps differ from the compiled ones
