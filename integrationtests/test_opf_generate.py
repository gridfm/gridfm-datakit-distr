"""``generate`` with OPF labels on catalog networks: a CSIRO feeder and IEEE 13."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.writer import DatasetWriter
from gridfm_datakit_distr.opf.problem import SOLVED
from gridfm_datakit_distr.solve.executor import generate

pytestmark = [pytest.mark.network_data]

CONFIG: dict[str, Any] = {
    "dataset_name": "opf",
    "networks": {"include": ["csiro_enwl/n02_f5", "ieee13"]},
    "variants": {
        "n_variants": 1,
        "admittance": {"line_length_scale": 1.0},
        "phase_reallocation": {"enabled": False},
    },  # unperturbed, as before the defaults
    # 40 kWp at every customer, inverters rated at their kWp: some scenarios curtail
    "der": {"pv": {"penetration": 1.0, "kwp": 40.0, "inverter_kva_per_kwp": 1.0}},
    "scenarios": {"n_scenarios_per_variant": 6, "pv_profile": {"cloudiness": 0.0}},
    "labels": {"opf": {"enabled": True}},
    "settings": {"seed": 2, "num_processes": 1},  # two of the six scenarios curtail
}


def rows_of(out: Path, network_id: str) -> tuple[list[dict[str, Any]], set[int]]:
    writer = DatasetWriter(out)
    meta = writer.read("scenarios/scenario_meta")
    opf = writer.read("opf/opf_meta")
    assert meta is not None
    assert opf is not None
    ids = set(pc.filter(meta, pc.equal(meta["network_id"], network_id))["scenario_id"].to_pylist())
    return [r for r in opf.to_pylist() if r["scenario_id"] in ids], ids


def test_generate_writes_verified_opf_labels(
    network_master: Callable[[str], Path], tmp_path: Path
) -> None:
    network_master("csiro_enwl/n02_f5")
    network_master("ieee13")
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(CONFIG), out)
    for name in ("opf/opf_meta", "opf/opf_der", "opf/opf_node_results"):
        parts = sorted((out / name).glob("part-*.parquet"))
        assert parts, name
        for part in parts:
            schema.check_schema(pq.read_table(part), schema.SCENARIO_TABLES[name]("mixed"), name)

    rows, ids = rows_of(out, "csiro_enwl/n02_f5")
    assert len(rows) == len(ids) == 6
    assert {r["opf_status"] for r in rows} == {SOLVED}
    assert all(r["verified"] for r in rows)
    assert any(r["opf_class"] == "B" and r["curtailment_w"] > 1.0 for r in rows)

    # IEEE 13: transformers, regulators at the scenario's taps, delta and phase-to-phase loads
    rows, ids = rows_of(out, "ieee13")
    assert len(rows) == len(ids) == 6
    # no_free_variable: a class-B scenario at night, every PV unit without available power
    assert {r["opf_status"] for r in rows} <= {
        SOLVED,
        "Infeasible_Problem_Detected",
        "no_free_variable",
    }
    assert all(r["verified"] for r in rows if r["opf_status"] == SOLVED)
    assert sum(r["opf_status"] == SOLVED for r in rows) >= 4
    nodes = DatasetWriter(out).read("scenarios/node_results", columns=["scenario_id"])
    assert nodes is not None
    assert ids <= set(nodes.column("scenario_id").to_pylist())  # power-flow labels kept
