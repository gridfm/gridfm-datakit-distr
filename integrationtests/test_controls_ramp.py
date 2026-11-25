"""The load-ramp fallback on IEEE 34: a scenario whose settling fails at the initial taps."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from gridfm_datakit_distr.config.load import load_config
from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.solve import executor

pytestmark = [pytest.mark.network_data]

REPO = Path(__file__).resolve().parents[1]


def test_a_scenario_unreachable_at_the_initial_taps_converges_after_the_ramp(
    network_master: Callable[[str], Path],
) -> None:
    """Scenario 27 of variant 0 (seed 42, load scale U(0.2, 1.2)): 1.28 times the nominal load; the
    first settling solve at taps 1.0 does not converge, the ramp lets the regulators rise."""
    network_master("ieee34")
    data = load_config(REPO / "scripts" / "config" / "health.yaml").model_dump(mode="json")
    data["networks"]["include"] = ["ieee34"]
    data["scenarios"]["profiles"]["synthetic"]["global_scale"] = {  # the scale it was chosen at
        "distribution": "uniform",
        "low": 0.2,
        "high": 1.2,
    }
    data["scenarios"]["profiles"]["synthetic"]["variant_scale"] = 1.0  # and no load level
    data["settings"].update(num_processes=1)
    cfg = GenerationConfig.model_validate(data)
    job = executor.jobs_of(cfg, load_catalog())[0]
    slot = executor._slot(0, executor.Task(job, 0, ()), cfg, None)
    assert slot.runner is not None
    o = slot.runner.run(27)
    assert o.pf_status == "converged"
    assert o.n_off_setpoint == 0
    assert o.origin is not None
    assert "ramp" in o.origin
