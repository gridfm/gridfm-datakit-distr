"""OPF labels in ``generate`` on inline feeders (needs the ``opf`` extra)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.manifest import Manifest
from gridfm_datakit_distr.io.writer import DatasetWriter
from gridfm_datakit_distr.opf.problem import SOLVED
from gridfm_datakit_distr.solve.executor import generate
from tests.circuits import opf_catalog as catalog
from tests.test_executor import same_dataset

CONFIG: dict[str, Any] = {
    "dataset_name": "opf",
    "networks": {"include": ["t/*"]},
    # inverters rated at their kWp: with headroom the reactive power alone clears the overvoltage
    "der": {"pv": {"penetration": 1.0, "kwp": 30.0, "inverter_kva_per_kwp": 1.0}},
    "scenarios": {
        "n_scenarios_per_variant": 8,
        "pv_profile": {"cloudiness": 0.0},
    },
    "labels": {"opf": {"enabled": True}},
    "settings": {"seed": 4, "chunk_size": 4, "num_processes": 1},
}


def config(**changes: Any) -> GenerationConfig:
    data = copy.deepcopy(CONFIG)
    for path, value in changes.items():
        node = data
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return GenerationConfig.model_validate(data)


def run_generate(root: Path, out: Path, data: dict[str, Any]) -> bool:
    """``generate`` in a fresh process; whether CasADi was imported in it. The solver's threads
    are not pinned: setting ``OPENBLAS_NUM_THREADS`` with CasADi loaded crashes OpenDSS contexts
    created in threads, and on this feeder pinned and free threads give the same tables bit for
    bit."""
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from gridfm_datakit_distr.config.schema import GenerationConfig\n"
        "from gridfm_datakit_distr.solve.executor import generate\n"
        "from tests.circuits import opf_catalog as catalog\n"
        "root, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "config = GenerationConfig.model_validate(json.loads(sys.argv[3]))\n"
        "generate(config, out, catalog=catalog(root), root=root)\n"
        "print('casadi' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(root), str(out), json.dumps(data)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1] == "True"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("opf_labels")
    out = root / "out"
    generate(config(), out, catalog=catalog(root), root=root)
    return out


def read(out: Path, table: str) -> Any:
    data = DatasetWriter(out).read(table)
    assert data is not None, table
    return data


def test_every_scenario_has_an_opf_row(dataset: Path) -> None:
    meta = read(dataset, "scenarios/scenario_meta")
    opf = read(dataset, "opf/opf_meta")
    assert sorted(opf.column("scenario_id").to_pylist()) == sorted(
        meta.column("scenario_id").to_pylist()
    )
    for name in ("opf/opf_meta", "opf/opf_der", "opf/opf_node_results"):
        for part in sorted((dataset / name).glob("part-*.parquet")):
            schema.check_schema(pq.read_table(part), schema.SCENARIO_TABLES[name]("mixed"), name)


def test_the_slice_feeder_is_solved_and_verified(dataset: Path) -> None:
    meta = read(dataset, "scenarios/scenario_meta")
    lv = set(pc.filter(meta, pc.equal(meta["network_id"], "t/lv"))["scenario_id"].to_pylist())
    opf = read(dataset, "opf/opf_meta").to_pylist()
    rows = [r for r in opf if r["scenario_id"] in lv]
    assert rows
    assert {r["opf_status"] for r in rows} == {SOLVED}
    assert all(r["verified"] and r["verify_reason"] is None for r in rows)
    assert all(r["verify_max_dv_pu"] <= 1e-6 for r in rows)
    assert all((r["cost_preset"], r["source_c1"], r["source_c2"]) == ("economic", 1.0, 0.0)
               for r in rows)  # fmt: skip
    assert {r["opf_class"] for r in rows} == {"A", "B"}
    assert any(r["opf_class"] == "B" and r["curtailment_w"] > 1.0 for r in rows)
    mine = pa.array(sorted(lv), pa.int64())
    der = read(dataset, "opf/opf_der")
    der = der.filter(pc.is_in(der["scenario_id"], mine))
    assert der.num_rows == 3 * len(rows)  # one PV unit per customer
    assert set(der.column("c1").to_pylist()) == set(der.column("c2").to_pylist()) == {0.0}
    nodes = read(dataset, "opf/opf_node_results")
    assert nodes.filter(pc.is_in(nodes["scenario_id"], mine)).num_rows == 12 * len(rows)
    engines = {json.loads(e)["casadi"] for e in meta.column("engine_versions").to_pylist()}
    assert len(engines) == 1


def test_a_network_with_a_transformer_is_solved_and_verified(dataset: Path) -> None:
    meta = read(dataset, "scenarios/scenario_meta")
    mvlv = set(pc.filter(meta, pc.equal(meta["network_id"], "t/mvlv"))["scenario_id"].to_pylist())
    rows = [r for r in read(dataset, "opf/opf_meta").to_pylist() if r["scenario_id"] in mvlv]
    assert len(rows) == len(mvlv) == 8
    assert {r["opf_status"] for r in rows} == {SOLVED}
    assert all(r["verified"] for r in rows)


def test_a_network_outside_the_scope_keeps_its_power_flow_labels(dataset: Path) -> None:
    meta = read(dataset, "scenarios/scenario_meta")
    other = pc.filter(meta, pc.equal(meta["network_id"], "t/other"))
    ids = set(other["scenario_id"].to_pylist())
    assert set(other["pf_status"].to_pylist()) == {"converged"}
    pf_nodes = read(dataset, "scenarios/node_results")
    assert ids <= set(pf_nodes.column("scenario_id").to_pylist())
    opf = read(dataset, "opf/opf_meta").to_pylist()
    rows = [r for r in opf if r["scenario_id"] in ids]
    assert len(rows) == len(ids)
    assert {r["opf_status"] for r in rows} == {"not_supported"}
    assert all("pvsystem not covered by the OPF yet" in r["opf_message"] for r in rows)
    assert all(r["objective"] is None and r["verified"] is None for r in rows)
    counts = Counter(r["opf_status"] for r in opf)
    card = (dataset / "README.md").read_text(encoding="utf-8")
    listed = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    assert f"- OPF labels: {listed}" in card
    networks = {n["network_id"]: n for n in Manifest.load(dataset).networks}
    assert networks["t/lv"]["opf_not_supported"] is None
    assert networks["t/mvlv"]["opf_not_supported"] is None
    assert "PVSystem.pv" in networks["t/other"]["opf_not_supported"]


def test_a_variant_without_pv_is_solved(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = config(**{"der.pv.penetration": 0.0, "networks.include": ["t/lv"]})
    generate(data, out, catalog=catalog(tmp_path), root=tmp_path)
    opf = read(out, "opf/opf_meta").to_pylist()
    assert {r["opf_status"] for r in opf} == {SOLVED}
    assert all(r["verified"] and r["curtailment_w"] == 0.0 for r in opf)
    assert read(out, "opf/opf_der").num_rows == 0
    # with nothing to dispatch, the objective is the power the source delivers in the power flow
    assert all(np.isfinite(r["objective"]) and r["objective"] > 0 for r in opf)


def test_without_opf_labels_casadi_is_not_imported(tmp_path: Path) -> None:
    data = config().model_dump(mode="json")
    data["labels"]["opf"]["enabled"] = False
    assert not run_generate(tmp_path, tmp_path / "out", data)
    assert not (tmp_path / "out" / "opf").exists()


def test_the_opf_tables_do_not_depend_on_the_process_count(tmp_path: Path) -> None:
    data = config(**{"settings.chunk_size": 2}).model_dump(mode="json")
    one, four = tmp_path / "one", tmp_path / "four"
    run_generate(tmp_path, one, data)
    data["settings"]["num_processes"] = 4
    run_generate(tmp_path, four, data)
    assert (one / "opf" / "opf_meta").is_dir()
    same_dataset(one, four)  # every table bit for bit, but the timing columns


def test_without_violations_every_scenario_is_class_a_at_full_output(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = config(**{
        "der.pv.kwp": 2.0, "networks.include": ["t/lv"], "settings.float_precision": "full",
    })  # fmt: skip
    generate(data, out, catalog=catalog(tmp_path), root=tmp_path)
    opf = read(out, "opf/opf_meta").to_pylist()
    assert {(r["opf_status"], r["opf_class"]) for r in opf} == {(SOLVED, "A")}
    avail = {
        (r["scenario_id"], r["element_idx"]): r["p_avail_w"]
        for r in read(out, "scenarios/der").to_pylist()
    }
    for r in read(out, "opf/opf_der").to_pylist():
        expected = avail[(r["scenario_id"], r["element_idx"])]
        assert r["p_w"] == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.fixture(scope="module")
def sampled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("opf_sampled")
    out = root / "out"
    data = config(**{
        "labels.opf.costs": "sampled", "networks.include": ["t/lv"],
        "settings.float_precision": "full",
    })  # fmt: skip
    generate(data, out, catalog=catalog(root), root=root)
    return out


def test_stored_costs_reproduce_the_objective(sampled: Path) -> None:
    meta = {r["scenario_id"]: r for r in read(sampled, "opf/opf_meta").to_pylist()}
    der = read(sampled, "opf/opf_der").to_pylist()
    for sid, m in meta.items():
        assert m["cost_preset"] == "sampled"
        mine = [r for r in der if r["scenario_id"] == sid]
        p0 = m["source_p_w"]
        cost = m["source_c1"] * p0 + m["source_c2"] * p0**2
        cost += sum(r["c1"] * r["p_w"] + r["c2"] * r["p_w"] ** 2 for r in mine)
        assert cost == pytest.approx(m["objective"], rel=1e-9, abs=1e-6)


def test_sampled_costs_differ_by_unit_and_scenario(sampled: Path) -> None:
    der = read(sampled, "opf/opf_der").to_pylist()
    c1 = [r["c1"] for r in der]
    assert len(set(c1)) == len(c1)  # one draw per unit and scenario
    assert all(0.0 <= c < 1.0 for c in c1)
    assert {r["source_c1"] for r in read(sampled, "opf/opf_meta").to_pylist()} == {1.0}


def test_active_limits_are_counted(dataset: Path) -> None:
    opf = read(dataset, "opf/opf_meta").to_pylist()
    solved = [r for r in opf if r["opf_status"] == SOLVED]
    assert all(r["n_vmin_active"] is not None for r in solved)
    curtailing = [r for r in solved if r["curtailment_w"] > 1.0]
    assert curtailing
    # PV curtailed for an overvoltage: an upper voltage bound (or a rating) binds
    assert all(r["n_vmax_active"] + r["n_rating_active"] >= 1 for r in curtailing)
    assert all(r["controls_hold"] is None for r in solved)  # no regulator in these networks


def test_controls_hold_is_reported_on_a_regulated_network(tmp_path: Path) -> None:
    from tests.test_executor import CONFIG as EXECUTOR_CONFIG
    from tests.test_executor import catalog as executor_catalog

    data = copy.deepcopy(EXECUTOR_CONFIG)
    data["labels"] = {"opf": {"enabled": True}}
    data["networks"] = {"include": ["t/reg"]}
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(data), out, catalog=executor_catalog(tmp_path),
             root=tmp_path)  # fmt: skip
    opf = read(out, "opf/opf_meta").to_pylist()
    solved = [r for r in opf if r["opf_status"] == SOLVED]
    assert solved
    assert all(r["verified"] for r in solved)
    assert all(isinstance(r["controls_hold"], bool) for r in solved)
