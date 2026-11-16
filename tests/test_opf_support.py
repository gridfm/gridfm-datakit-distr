"""Scope of the OPF problem, and the power-flow path free of CasADi."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.opf.support import unsupported_reason
from tests.circuits import ALL_CLASSES, FOUR_WIRE, MV_LV_RNEUT, OPF_FEEDER, write_circuit


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
    assert "generator that is not a PV unit: Generator.g" in reason
    assert "pvsystem not covered by the OPF yet: PVSystem.pv" in reason
    assert "storage not covered by the OPF yet: Storage.st" in reason


def test_the_power_flow_path_does_not_import_casadi() -> None:
    code = (
        "import sys\n"
        "import gridfm_datakit_distr.cli, gridfm_datakit_distr.solve.executor\n"
        "import gridfm_datakit_distr.solve.pf_batch, gridfm_datakit_distr.opf.support\n"
        "assert 'casadi' not in sys.modules, 'casadi imported'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
