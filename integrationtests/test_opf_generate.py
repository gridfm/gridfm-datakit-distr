"""``generate`` with OPF labels on catalog networks: a CSIRO feeder and IEEE 13."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import yaml

pytest.importorskip("casadi")

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.manifest import Manifest
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
    # the conditions these scenarios were chosen at: without a load level (at the default level
    # 1.64 IEEE 13's scenario 5 verifies at 1.3e-6 pu, above the 1e-6 of verification)
    "scenarios": {
        "n_scenarios_per_variant": 6,
        "pv_profile": {"cloudiness": 0.0},
        "profiles": {"synthetic": {"variant_scale": 1.0}},
    },
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


def test_storage_on_the_small_opf_config(
    network_master: Callable[[str], Path], tmp_path: Path
) -> None:
    """The getting-started OPF config with storage at half its PV customers: every scenario has
    an OPF row, the storage set-points enter the power flow, and every solved OPF verifies."""
    network_master("csiro_enwl/n02_f5")
    path = Path(__file__).resolve().parents[1] / "scripts/config/opf_small.yaml"
    data = yaml.safe_load(path.read_text())
    data["der"]["storage"] = {"share": 0.5}
    data["settings"]["num_processes"] = 1
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(data), out)
    writer = DatasetWriter(out)
    der = writer.read("scenarios/der")
    assert der is not None
    bess = pc.filter(der, pc.equal(der["der_type"], "bess"))
    assert bess.num_rows > 0
    assert pc.max(pc.abs(bess["p_w"])).as_py() > 1e3
    rows, ids = rows_of(out, "csiro_enwl/n02_f5")
    assert len(rows) == len(ids) == 10
    solved = [r for r in rows if r["opf_status"] == SOLVED]
    assert solved
    assert all(r["verified"] for r in solved)


def test_dg_on_the_small_opf_config(network_master: Callable[[str], Path], tmp_path: Path) -> None:
    """The getting-started OPF config with dispatchable generators: the feeder's capacity binds on
    the voltage bound, the DG enter the power flow at their set-points, and every solved OPF
    dispatches them and verifies."""
    network_master("csiro_enwl/n02_f5")
    path = Path(__file__).resolve().parents[1] / "scripts/config/opf_small.yaml"
    data = yaml.safe_load(path.read_text())
    data["der"]["dg"] = {"enabled": True}
    data["settings"]["num_processes"] = 1
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(data), out)
    (network,) = Manifest.load(out).networks
    assert (network["dg"]["status"], network["dg"]["binding"]) == ("ok", "voltage")
    writer = DatasetWriter(out)
    der = writer.read("scenarios/der")
    assert der is not None
    dg = pc.filter(der, pc.equal(der["der_type"], "dg"))
    assert dg.num_rows == 5 * 10  # every 4th of the 23 loads, ten scenarios
    dg_idx = set(dg["element_idx"].to_pylist())
    rows, ids = rows_of(out, "csiro_enwl/n02_f5")
    assert len(rows) == len(ids) == 10
    solved = [r for r in rows if r["opf_status"] == SOLVED]
    assert solved
    assert all(r["verified"] for r in solved)
    dispatch = writer.read("opf/opf_der")
    assert dispatch is not None
    mine = pc.filter(dispatch, pc.is_in(dispatch["element_idx"], pa.array(sorted(dg_idx))))
    assert mine.num_rows == 5 * len(solved)
    assert pc.max(mine["p_w"]).as_py() > 0
