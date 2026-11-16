"""Scope of the OPF problem, and the power-flow path free of CasADi."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.opf.support import unsupported_reason
from tests.circuits import FOUR_WIRE, MV_LV_RNEUT, OPF_FEEDER, write_circuit


def model_of(tmp_path: Path, circuit: dict[str, str]) -> ElectricalModel:
    write_circuit(tmp_path / "src", circuit)
    network = Network(id="t/opf", source="src", master="master.dss")
    return prepare_network(network, root=tmp_path).model


def test_the_slice_feeder_is_supported(tmp_path: Path) -> None:
    assert unsupported_reason(model_of(tmp_path, OPF_FEEDER)) is None


def test_a_transformer_is_not_supported(tmp_path: Path) -> None:
    reason = unsupported_reason(model_of(tmp_path, MV_LV_RNEUT))
    assert reason is not None
    assert "transformer not covered by the OPF yet: Transformer.t" in reason


def test_a_delta_load_is_not_supported(tmp_path: Path) -> None:
    reason = unsupported_reason(model_of(tmp_path, FOUR_WIRE))
    assert reason == "load not single-phase phase-to-neutral: Load.delta"


def test_the_power_flow_path_does_not_import_casadi() -> None:
    code = (
        "import sys\n"
        "import gridfm_datakit_distr.cli, gridfm_datakit_distr.solve.executor\n"
        "import gridfm_datakit_distr.solve.pf_batch, gridfm_datakit_distr.opf.support\n"
        "assert 'casadi' not in sys.modules, 'casadi imported'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
