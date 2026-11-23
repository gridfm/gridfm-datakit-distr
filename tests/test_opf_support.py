"""Scope of the OPF problem, and the power-flow path free of CasADi."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.ground import conventional_zones
from gridfm_datakit_distr.model.io_parquet import variant_level_tables
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.opf.support import floating_buses, unsupported_reason
from tests.circuits import (
    ALL_CLASSES,
    DG_FEEDER,
    FOUR_WIRE,
    MV_LV_FLOATING,
    MV_LV_FLOATING_CAPACITIVE,
    MV_LV_RNEUT,
    OPF_FEEDER,
    write_circuit,
)


def model_of(tmp_path: Path, circuit: dict[str, str]) -> ElectricalModel:
    write_circuit(tmp_path / "src", circuit)
    network = Network(id="t/opf", source="src", master="master.dss")
    return prepare_network(network, root=tmp_path).model


def test_the_slice_feeder_is_supported(tmp_path: Path) -> None:
    assert unsupported_reason(model_of(tmp_path, OPF_FEEDER)) is None


def test_transformers_and_delta_loads_are_covered(tmp_path: Path) -> None:
    assert unsupported_reason(model_of(tmp_path / "mvlv", MV_LV_RNEUT)) is None
    assert unsupported_reason(model_of(tmp_path / "delta", FOUR_WIRE)) is None


def test_pv_systems_storage_and_other_generators_are_not_covered(tmp_path: Path) -> None:
    reason = unsupported_reason(model_of(tmp_path, ALL_CLASSES))
    assert reason is not None
    assert "generator that is not a PV, storage or DG unit: Generator.g" in reason
    assert "pvsystem not covered by the OPF yet: PVSystem.pv" in reason
    assert "storage not covered by the OPF yet: Storage.st" in reason


def test_dispatchable_generators_are_covered(tmp_path: Path) -> None:
    dg = "New Generator.dg_h3 bus1=b1.3.4 phases=1 kV=0.23 kW=5 kvar=0 model=1\n"
    circuit = {"master.dss": DG_FEEDER["master.dss"].replace("Set Volt", dg + "Set Volt")}
    model = model_of(tmp_path, circuit)
    assert "Generator.dg_h3" in {e.name for e in model.elements}
    assert unsupported_reason(model) is None


def test_the_power_flow_path_does_not_import_casadi() -> None:
    code = (
        "import sys\n"
        "import gridfm_datakit_distr.cli, gridfm_datakit_distr.solve.executor\n"
        "import gridfm_datakit_distr.solve.pf_batch, gridfm_datakit_distr.opf.support\n"
        "import gridfm_datakit_distr.variants.dg\n"
        "assert 'casadi' not in sys.modules, 'casadi imported'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_zone_tied_to_ground_by_a_convention_only_is_covered_and_marked(tmp_path: Path) -> None:
    """Behind a transformer whose LV neutral is left open, only the windings' anti-float
    admittance ties the LV zone to ground: covered, its nodes marked in the stored nodes."""
    model = model_of(tmp_path, MV_LV_FLOATING)
    assert unsupported_reason(model) is None
    (zone,) = conventional_zones(model)
    assert {model.nodes[i].bus for i in zone} == {"lv", "c"}
    marked = variant_level_tables(model)["variants/nodes"].to_pylist()
    assert {r["bus"] for r in marked if r["ground_by_convention"]} == {"lv", "c"}


def test_a_zone_tied_to_ground_by_line_capacitance_only_is_not_covered(tmp_path: Path) -> None:
    model = model_of(tmp_path, MV_LV_FLOATING_CAPACITIVE)
    assert floating_buses(model) == ["c", "lv"]
    assert conventional_zones(model) == []
    reason = unsupported_reason(model)
    assert reason is not None
    assert reason.startswith("buses tied to ground only through line capacitance")
    assert "mv2" not in reason  # the MV zone holds the source
    nodes = variant_level_tables(model)["variants/nodes"].to_pylist()
    assert not any(r["ground_by_convention"] for r in nodes)  # physical, if weak
