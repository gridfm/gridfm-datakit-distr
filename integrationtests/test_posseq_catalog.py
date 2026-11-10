"""The positive-sequence export on catalog networks: the view, built from the stored truth by
``export-posseq``, meets datakit's power balance and stays close to the truth."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.views.posseq import power_balance
from gridfm_datakit_distr.views.posseq_export import export_posseq

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]

# Largest |V_view - V1_truth| (pu) measured on 6 scenarios (30 % PV), and the bound
# asserted here: the view is an approximation of an unbalanced network, its error is data.
GAP = {
    "ieee13": (2.1e-3, 0.01),
    "ieee37": (1.2e-2, 0.03),  # open-delta regulator with unequal taps
    "csiro_enwl/n01_f1": (1.1e-3, 0.01),
    "edp/industrial": (1.2e-8, 1e-6),  # balanced loads: the view is exact
}


@pytest.mark.parametrize("network_id", sorted(GAP))
def test_the_export_per_scenario(
    network_master: Callable[[str], Path], network_id: str, tmp_path: Path
) -> None:
    network_master(network_id)
    cfg = GenerationConfig.model_validate(
        {
            "dataset_name": "t",
            "networks": {"include": [network_id]},
            "scenarios": {"n_scenarios_per_variant": 4},
            "der": {"pv": {"penetration": 0.3}},
            "settings": {"seed": 3, "num_processes": 1},
        }
    )
    out = tmp_path / "d"
    generate(cfg, out)
    [result] = export_posseq(out, tmp_path / "x")
    assert result.scenarios > 0
    assert result.statuses == {"converged": result.scenarios}
    raw = result.raw_dir / "{}" / "scenario_partition=0"
    bus = pq.read_table(str(raw).format("bus_data.parquet"))
    branch = pq.read_table(str(raw).format("branch_data.parquet"))
    s_base_va = pq.read_table(out / "catalog" / "networks.parquet").column("s_base_va")[0].as_py()
    balance = power_balance(bus, branch)
    assert balance.max() < 1e-6 * s_base_va / 1e6  # datakit's power balance
    v = bus.column("Vm").to_numpy() * np.exp(1j * np.radians(bus.column("Va").to_numpy()))
    truth = bus.column("x_truth_Vm").to_numpy() * np.exp(
        1j * np.radians(bus.column("x_truth_Va").to_numpy())
    )
    assert np.abs(v - truth).max() < GAP[network_id][1]
