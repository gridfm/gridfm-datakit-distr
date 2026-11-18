"""Closed loop on catalog networks: the stored tables alone reproduce the stored voltages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io.reader import DatasetReader
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.validate.closed_loop import closed_loop

pytestmark = [pytest.mark.conformance, pytest.mark.network_data, pytest.mark.slow]

# One tolerance at both precisions: with float64 everywhere the stored OpenDSS solution of a
# stiff network (near-ideal switches, weakly grounded zones) is itself exact only to about 1e-8 pu.
TOLERANCE_PU = 1e-6
NETWORKS = [
    "ieee13",
    "ieee34",
    "ieee37",
    "ieee123",
    "ieee_eulv",
    "ieee8500",
    "csiro_enwl/n14_f6",
    "csiro_enwl/n21_f3",
    "edp/industrial",
    "edp/rural",
]


def check(
    network_master: Callable[[str], Path], tmp: Path, network_id: str, precision: str
) -> None:
    network_master(network_id)
    cfg = GenerationConfig.model_validate(
        {
            "dataset_name": "loop",
            "networks": {"include": [network_id]},
            "variants": {"n_variants": 1},
            "der": {"pv": {"penetration": 0.5}},
            "scenarios": {"n_scenarios_per_variant": 3},
            "settings": {"seed": 5, "num_processes": 1, "float_precision": precision},
        }
    )
    generate(cfg, tmp / "out")
    reader = DatasetReader(tmp / "out")
    ids = [r["scenario_id"] for r in reader.meta.to_pylist() if r["pf_status"] == "converged"]
    assert ids
    worst = max(closed_loop(sc).max_dv_pu for sc in reader.scenarios(ids))
    print(f"{network_id} {precision}: {worst:.1e} pu")  # the measured value, shown by `pytest -rA`
    assert worst <= TOLERANCE_PU, worst


@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("network_id", NETWORKS)
def test_the_closed_loop(
    network_master: Callable[[str], Path], tmp_path: Path, network_id: str, precision: str
) -> None:
    check(network_master, tmp_path, network_id, precision)
