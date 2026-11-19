"""Tests for the schema module, the writer, the manifest and the chunk tables."""

from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr import __version__
from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.card import dataset_card
from gridfm_datakit_distr.io.manifest import Manifest, Unit, config_hash
from gridfm_datakit_distr.io.writer import STATES_PART, DatasetWriter, part_key
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.solve.tables import chunk_tables, state_tables, variant_tables
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from tests.circuits import CONTROLLED, VARIANT_FEEDER, write_circuit

REPO = Path(__file__).resolve().parents[1]
CONFIG: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["x"]},
    "scenarios": {"n_scenarios_per_variant": 6},
    "settings": {"seed": 3, "residual_sample_rate": 1.0},
}


# ------------------------------------------------------------------- schema = SCHEMA.md


def documented_columns() -> dict[str, list[str]]:
    """Column names of every table documented with a column table in docs/manual/outputs.md."""
    text = (REPO / "docs" / "manual" / "outputs.md").read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    for section in re.split(r"^### ", text, flags=re.M)[1:]:
        head = section.splitlines()[0]
        m = re.match(r"`([a-z_/]+?)(?:\.parquet|/part-\*\.parquet)?`", head)
        if not m:
            continue
        columns = []
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("| `") and len(cells) >= 2:
                columns += re.findall(r"`([A-Za-z_0-9]+)`", cells[0])
        if columns:
            out[m.group(1)] = columns
    return out


def test_every_documented_table_has_its_schema() -> None:
    documented = documented_columns()
    assert {"scenarios/scenario_meta", "scenarios/node_results", "catalog/networks"} <= set(
        documented
    )
    for name, columns in documented.items():
        if name in schema.SCENARIO_TABLES:
            actual = schema.SCENARIO_TABLES[name]("mixed").names
        elif name in schema.TABLES:
            actual = schema.TABLES[name].names
        else:
            pytest.fail(f"{name} is documented but has no schema")
        assert actual == columns, name


def test_precision() -> None:
    mixed, full = schema.node_results("mixed"), schema.node_results("full")
    assert mixed.field("v_re").type == full.field("v_re").type == pa.float64()
    mixed, full = schema.element_results("mixed"), schema.element_results("full")
    assert mixed.field("i_re").type == pa.float32()
    assert full.field("i_re").type == pa.float64()


# ------------------------------------------------------------------- writer and manifest


def test_writes_are_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = DatasetWriter(tmp_path)
    table = pa.table({"state_id": ["s"], "row": [0], "col": [0], "g": [1.0], "b": [0.0]})
    table = table.cast(schema.YBUS_STATE)
    rel = writer.write_part("states/ybus", part_key(0, 1, 2), table, schema.YBUS_STATE)
    assert rel == "states/ybus/part-00000-0001-00002.parquet"
    assert pq.read_table(tmp_path / rel).equals(table)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        writer.write_part("states/ybus", part_key(0, 1, 3), table, schema.YBUS_STATE)
    assert sorted(p.name for p in (tmp_path / "states" / "ybus").iterdir()) == [Path(rel).name]
    with pytest.raises(ValueError, match="does not match its schema"):
        writer.write_part("states/ybus", "x", pa.table({"a": [1]}), schema.YBUS_STATE)


def test_encodings(tmp_path: Path) -> None:
    """Dictionaries only for strings (most floats are distinct), delta for the integer keys."""
    table = pa.table(
        {
            "scenario_id": pa.array([5, 5, 6, 6], pa.int64()),
            "node_idx": pa.array([0, 1, 0, 1], pa.int32()),
            "v_re": [1.0, 2.0, 3.0, 4.0],
            "bus": ["a", "b", "a", "b"],
        }
    )
    DatasetWriter(tmp_path).write_table("t", table, table.schema)
    meta = pq.ParquetFile(tmp_path / "t.parquet").metadata.row_group(0)
    encodings = {meta.column(i).path_in_schema: meta.column(i).encodings for i in range(4)}
    assert "DELTA_BINARY_PACKED" in encodings["scenario_id"]
    assert "DELTA_BINARY_PACKED" in encodings["node_idx"]
    assert "RLE_DICTIONARY" not in encodings["v_re"]
    assert "RLE_DICTIONARY" in encodings["bus"]
    assert pq.read_table(tmp_path / "t.parquet").equals(table)


def test_manifest(tmp_path: Path) -> None:
    config = GenerationConfig.model_validate(CONFIG).model_dump(mode="json")
    manifest = Manifest.create(tmp_path, config, {"generator": __version__, "opendss": "x"})
    unit = Unit("t/net", 0, 0, 0, 0, 6)
    manifest.add_units([unit])
    manifest.finish(unit, "done")
    back = Manifest.load(tmp_path)
    assert back.config_hash == manifest.config_hash
    assert back.units[unit.key].status == "done"
    # the hash ignores execution settings, not the data's
    other = copy.deepcopy(config)
    other["settings"]["num_processes"] = 32
    assert config_hash(other) == manifest.config_hash
    other["settings"]["seed"] = 4
    assert config_hash(other) != manifest.config_hash


# ------------------------------------------------------------------- chunk tables


def run_chunk(
    tmp_path: Path, circuit: dict[str, str], **sections: Any
) -> tuple[VariantRunner, Any, list[Any]]:
    data = copy.deepcopy(CONFIG)
    data.update(sections)
    cfg = GenerationConfig.model_validate(data)
    write_circuit(tmp_path / "src", circuit)
    network = Network(id="t/net", source="src", master="master.dss")
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options, tmp_path))
    result = generate_variant(
        network, targets, cfg.variants, 3, 0, options=options, root=tmp_path, der=cfg.der.pv
    )
    runner = VariantRunner(network, result, cfg, 0, root=tmp_path)
    return runner, result, [runner.run(s) for s in range(6)]


@pytest.mark.parametrize("precision", ["mixed", "full"])
def test_chunk_tables(tmp_path: Path, precision: schema.Precision) -> None:
    runner, result, outcomes = run_chunk(
        tmp_path, VARIANT_FEEDER, der={"pv": {"penetration": 1.0}}, controls={"mode": "as_is"},
        labels={"element_currents": True},
    )  # fmt: skip
    versions = {"generator": __version__, "opendss": "x"}
    tables = chunk_tables(runner, outcomes, precision=precision, config_hash="h", versions=versions)
    for name, table in tables.items():
        schema.check_schema(table, schema.SCENARIO_TABLES[name](precision), name)
    converged = sum(o.pf_status == "converged" for o in outcomes)
    assert tables["scenarios/scenario_meta"].num_rows == 6
    meta = tables["scenarios/scenario_meta"].select(["state_id", "origin"]).to_pydict()
    assert meta["origin"] == [None if s is None else "as_is" for s in meta["state_id"]]
    assert tables["scenarios/node_results"].num_rows == converged * len(runner.model.nodes)
    assert tables["scenarios/element_results"].num_rows == converged * runner.plan.element_idx.size
    assert tables["scenarios/loads"].num_rows == 6 * len(runner.customers)
    assert tables["scenarios/der"].num_rows == 6 * 6
    v = variant_tables(result)
    for name, table in v.items():
        schema.check_schema(table, schema.TABLES[name], name)


def test_states_are_consolidated_with_their_counts(tmp_path: Path) -> None:
    runner, _, outcomes = run_chunk(tmp_path / "net", CONTROLLED, controls={"tap_jitter_steps": 1})
    writer = DatasetWriter(tmp_path / "data")
    versions = {"generator": __version__, "opendss": "x"}
    new = runner.new_states()
    assert new
    half = dict(list(new.items())[len(new) // 2 :])
    meta = chunk_tables(runner, outcomes, precision="mixed", config_hash="h", versions=versions)

    def write_chunk(writer: DatasetWriter, chunk: int, met: dict[str, Any]) -> None:
        for name, table in state_tables(met).items():
            expected = STATES_PART if name == "states/states" else schema.TABLES[name]
            writer.write_part(name, part_key(0, 0, chunk), table, expected)
        if chunk == 0:
            writer.write_part(
                "scenarios/scenario_meta",
                part_key(0, 0, 0),
                meta["scenarios/scenario_meta"],
                schema.scenario_meta("mixed"),
            )

    # two chunks that met overlapping states
    for chunk, met in ((0, new), (1, half)):
        write_chunk(writer, chunk, met)
    (tmp_path / "data" / "states" / "ybus" / ".part-x.parquet.tmp").write_bytes(b"killed")
    writer.consolidate_states()
    # the same tables when chunk 1 was consolidated first and chunk 0 came on resume
    resumed = DatasetWriter(tmp_path / "resumed")
    write_chunk(resumed, 1, half)
    resumed.consolidate_states()
    write_chunk(resumed, 0, new)
    resumed.consolidate_states()
    for name in ("states/states", "states/ybus", "states/yprim_state"):
        a = pq.read_table(tmp_path / "data" / f"{name}.parquet")
        assert a.equals(pq.read_table(tmp_path / "resumed" / f"{name}.parquet")), name
    states = pq.read_table(tmp_path / "data" / "states" / "states.parquet")
    schema.check_schema(states, schema.STATES, "states/states")
    assert sorted(states.column("state_id").to_pylist()) == sorted(new)  # each state once
    counts = Counter(o.state_id for o in outcomes if o.state_id)
    assert (
        dict(
            zip(
                states.column("state_id").to_pylist(),
                states.column("n_scenarios").to_pylist(),
                strict=True,
            )
        )
        == counts
    )
    ybus = pq.read_table(tmp_path / "data" / "states" / "ybus.parquet")
    assert set(ybus.column("state_id").to_pylist()) == set(new)
    assert not (tmp_path / "data" / "states" / "ybus").exists()


def test_consolidation_checks_each_part_against_its_own_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many small parts, each meeting states of its neighbours: each state's rows from the first
    part holding it, ordered by state; the membership test sees only each part's own states
    (against every state seen so far it grows with the number of parts)."""
    writer = DatasetWriter(tmp_path)
    n_parts, expected = 60, {}
    for c in range(n_parts):
        sids = [f"{j:016x}" for j in range(2 * c, 2 * c + 4)]  # two states shared with the next
        rows = [(s, r, float(c)) for s in sids for r in range(3)]
        for s, r, g in rows:
            expected.setdefault((s, r), g)
        table = pa.table(
            {
                "state_id": [s for s, _, _ in rows],
                "row": [r for _, r, _ in rows],
                "col": [0] * len(rows),
                "g": [g for _, _, g in rows],
                "b": [0.0] * len(rows),
            }
        ).cast(schema.YBUS_STATE)
        writer.write_part("states/ybus", part_key(0, 0, c), table, schema.YBUS_STATE)
    sizes: list[int] = []
    real = pc.is_in

    def counting(values: Any, value_set: Any, **kwargs: Any) -> Any:
        sizes.append(len(value_set))
        return real(values, value_set=value_set, **kwargs)

    monkeypatch.setattr(pc, "is_in", counting)
    writer.consolidate_states()
    got = pq.read_table(tmp_path / "states" / "ybus.parquet").to_pylist()
    assert [(r["state_id"], r["row"], r["g"]) for r in got] == [
        (s, r, g) for (s, r), g in sorted(expected.items())
    ]
    assert len(sizes) == n_parts
    assert max(sizes) <= 4  # each part's own states, never all those seen before


def test_dataset_card(tmp_path: Path) -> None:
    config = GenerationConfig.model_validate(CONFIG).model_dump(mode="json")
    manifest = Manifest.create(tmp_path, config, {"generator": __version__})
    passport = {
        "network_id": "ieee13", "source": "epri_ieee", "license": "BSD-3-Clause",
        "counts": {"nodes": 41}, "neutral_representation": "kron", "citation": "IEEE PES",
    }  # fmt: skip
    text = dataset_card(manifest, [passport], Counter({"converged": 5, "not_converged": 1}))
    assert text.startswith("# t\n")
    assert "| `ieee13` | epri_ieee | BSD-3-Clause | 41 | kron |" in text
    assert "Scenarios: 6 (converged 5, not_converged 1)" in text
    assert "- IEEE PES" in text
