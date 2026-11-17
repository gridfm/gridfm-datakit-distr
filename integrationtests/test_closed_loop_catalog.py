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

TOLERANCE = {"full": 1e-9, "mixed": 1e-6}  # mixed: float32 setpoints
FLOOR = {  # full precision: the stored OpenDSS solution is the limit, not the closed loop
    "ieee13": "3.6e-9 pu; the stored solution is 1.6e-9 to 2.5e-9 pu from the exact solution "
    "of the stored equations (switch of 1e7 S, per-unit condition number 2e8)",
    "ieee37": "5.2e-9 pu in the floating zone (voltages to ground undetermined), 3.4e-10 outside",
    "ieee123": "6.3e-9 pu at the floating bus 610, 1.7e-9 outside (switches of 1e6 S)",
    "edp/industrial": "2.1e-8 pu; KCL residual of the stored solution 2.8e-6 A, of the closed "
    "loop's 8.9e-7 A",
}
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
CASES = [
    pytest.param(
        n,
        p,
        marks=pytest.mark.xfail(strict=True, reason=f"owner to decide: {FLOOR[n]}")
        if p == "full" and n in FLOOR
        else (),
    )
    for n in NETWORKS
    for p in ("full", "mixed")
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
    assert worst <= TOLERANCE[precision], worst


@pytest.mark.parametrize(("network_id", "precision"), CASES)
def test_the_closed_loop(
    network_master: Callable[[str], Path], tmp_path: Path, network_id: str, precision: str
) -> None:
    check(network_master, tmp_path, network_id, precision)
