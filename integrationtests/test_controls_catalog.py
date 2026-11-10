"""Controls on catalog networks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.scenarios.controls import (
    apply_state,
    read_state,
    regulated_windings,
    settle,
)
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]


def scale_loads(p: PreparedNetwork, factor: float) -> None:
    api = p.ctx.api
    loads = []
    i = api.Loads.First()
    while i:
        loads.append((api.Loads.Name(), api.Loads.kW(), api.Loads.kvar()))
        i = api.Loads.Next()
    for name, kw, kvar in loads:
        p.ctx.command(f"Edit Load.{name} kW={kw * factor!r} kvar={kvar * factor!r}")


def min_phase_pu(p: PreparedNetwork) -> float:
    v = np.abs(p.ctx.ynode_voltages())
    base = np.array([n.kv_base_ln for n in p.model.nodes]) * 1e3
    phase = np.array([n.conductor in ("a", "b", "c") for n in p.model.nodes])
    return float((v / base)[phase].min())


@pytest.mark.parametrize("network_id", ["ieee13", "ieee123"])
def test_settling_and_jitter(network_master: Callable[[str], Path], network_id: str) -> None:
    network_master(network_id)
    p = prepare_network(load_catalog().network(network_id))
    vid = f"{network_id}/v0000"
    compiled = read_state(p.ctx, vid, "as_is")
    scale_loads(p, 1.5)
    settled, status = settle(p.ctx, "static_settle", vid)
    assert status is not None
    assert status.converged
    assert settled.taps != compiled.taps  # heavy load: the regulators move
    assert settle(p.ctx, "static_settle", vid)[0].state_id == settled.state_id

    limits = {
        (t, w): (lo, step, round((hi - lo) / step))
        for t, w, lo, hi, step in regulated_windings(p.ctx)
    }
    base = dict(settled.taps)
    seen = set()
    for s in range(30):
        apply_state(p.ctx, settled)
        rng = scenario_rng(7, 0, 0, s, ScenarioStream.TAP_JITTER)
        state, _ = settle(p.ctx, "static_settle", vid, jitter_steps=2, rng=rng)
        for t, taps in state.taps:
            for w, tap in enumerate(taps, start=1):
                if (t, w) not in limits:
                    assert tap == base[t][w - 1]  # unregulated windings stay
                    continue
                lo, step, n = limits[(t, w)]
                position = (tap - lo) / step
                assert position == pytest.approx(round(position), abs=1e-9)
                assert 0 <= round(position) <= n
        # the controls hold the state: settling from it moves nothing
        assert settle(p.ctx, "static_settle", vid)[0].state_id == state.state_id
        seen.add(state.state_id)
    assert len(seen) > 5


@pytest.mark.parametrize("network_id", ["ieee34", "ieee8500"])
def test_settling_lifts_the_compiled_base_case(
    network_master: Callable[[str], Path], network_id: str
) -> None:
    network_master(network_id)
    p = prepare_network(load_catalog().network(network_id))
    assert min_phase_pu(p) < 0.75  # taps as compiled
    _, status = settle(p.ctx, "static_settle", f"{network_id}/v0000")
    assert status is not None
    assert status.converged
    assert p.ctx.solve().converged  # the frozen state
    assert min_phase_pu(p) > 0.9
