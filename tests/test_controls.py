"""Tests for controls and network states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.state import NetworkState
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.scenarios.controls import (
    StateRegistry,
    apply_state,
    model_at_state,
    read_state,
    regulated_windings,
    settle,
)
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng
from tests.circuits import CONTROLLED, write_circuit

VID = "t/ctrl/v0000"


@pytest.fixture
def prepared(tmp_path: Path) -> PreparedNetwork:
    write_circuit(tmp_path / "src", CONTROLLED)
    return prepare_network(Network(id="t/ctrl", source="src", master="master.dss"), root=tmp_path)


def heavy(p: PreparedNetwork) -> None:
    p.ctx.command("Edit Load.ld kW=900 kvar=300")


# ------------------------------------------------------------------- states


def test_state_as_compiled(prepared: PreparedNetwork) -> None:
    state = read_state(prepared.ctx, VID, "as_is")
    assert state.taps == (("reg", (1.0, 1.0)),)
    assert state.capacitors == (("cap", (1,)),)
    assert len(state.state_id) == 16
    assert json.loads(state.taps_json()) == {"reg": [1.0, 1.0]}
    assert json.loads(state.capacitors_json()) == {"cap": [1]}


def test_state_ids_are_content_hashes() -> None:
    a = NetworkState(VID, (("reg", (1.0, 1.0125)),), (("cap", (1,)),), "static_settle")
    same = NetworkState(VID, (("reg", (1.0, 1.0125 + 1e-13)),), (("cap", (1,)),), "as_is")
    assert a.state_id == same.state_id  # rounded taps; the origin is not content
    assert (
        a.state_id
        != NetworkState(VID, (("reg", (1.0, 1.01875)),), (("cap", (1,)),), "as_is").state_id
    )
    assert (
        a.state_id
        != NetworkState(VID, (("reg", (1.0, 1.0125)),), (("cap", (0,)),), "as_is").state_id
    )
    assert a.state_id != NetworkState("t/ctrl/v0001", a.taps, a.capacitors, "as_is").state_id


def test_one_state_has_one_set_of_taps() -> None:
    """Settling and jitter reach one tap position an ulp apart (ieee13: 1.0124999999999997 and
    1.0125000000000002); the state stores, and the label solve applies, the same taps."""
    a = NetworkState(VID, (("reg", (1.0, 1.0124999999999997)),), (), "static_settle")
    b = NetworkState(VID, (("reg", (1.0, 1.0125000000000002)),), (), "static_settle+jitter")
    assert a.taps == b.taps == (("reg", (1.0, 1.0125)),)
    assert a.taps_json() == b.taps_json()


# ------------------------------------------------------------------- settling


def test_static_settling_moves_the_taps_under_heavy_load(prepared: PreparedNetwork) -> None:
    heavy(prepared)
    compiled = read_state(prepared.ctx, VID, "as_is")
    state, status = settle(prepared.ctx, "static_settle", VID, jitter_steps=0)
    assert status.converged
    assert state.origin == "static_settle"
    assert state.taps[0][1][1] > compiled.taps[0][1][1]  # the regulator raised its tap
    # the controls are frozen at the settled state: a new solve keeps it
    prepared.ctx.command("Edit Load.ld kW=100 kvar=30")
    assert prepared.ctx.solve().converged
    assert read_state(prepared.ctx, VID, "as_is").state_id == state.state_id


def test_as_is_keeps_the_compiled_taps(prepared: PreparedNetwork) -> None:
    heavy(prepared)
    state, status = settle(prepared.ctx, "as_is", VID)
    assert status is None
    assert state.origin == "as_is"
    assert state.taps == (("reg", (1.0, 1.0)),)


def test_the_controls_settle_again_from_the_jittered_taps(prepared: PreparedNetwork) -> None:
    """Every state is one the controls hold; starts on either side of the band stop at different
    positions inside it."""
    heavy(prepared)
    settled, _ = settle(prepared.ctx, "static_settle", VID, jitter_steps=0)
    ((t, w, lo, hi, step),) = regulated_windings(prepared.ctx)
    assert (t, w, lo, hi, step) == ("reg", 2, 0.9, 1.1, pytest.approx(0.00625))
    seen = set()
    for s in range(60):
        apply_state(prepared.ctx, settled)
        rng = scenario_rng(1, 0, 0, s, ScenarioStream.TAP_JITTER)
        state, status = settle(prepared.ctx, "static_settle", VID, jitter_steps=2, rng=rng)
        assert status is not None
        assert status.converged
        position = (state.taps[0][1][1] - lo) / step
        assert position == pytest.approx(round(position), abs=1e-9)
        seen.add(round(position))
        assert state.origin == (
            "static_settle+jitter" if state.state_id != settled.state_id else "static_settle"
        )
        # settling from the frozen state moves nothing
        assert settle(prepared.ctx, "static_settle", VID)[0].state_id == state.state_id
    assert len(seen) > 1


def test_a_frozen_state_reproduces_the_settled_solution(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", CONTROLLED)
    network = Network(id="t/ctrl", source="src", master="master.dss")
    a = prepare_network(network, root=tmp_path)
    heavy(a)
    state, _ = settle(a.ctx, "static_settle", VID)
    assert a.ctx.solve().converged
    b = prepare_network(network, root=tmp_path)
    heavy(b)
    apply_state(b.ctx, state)
    assert b.ctx.solve().converged
    np.testing.assert_allclose(b.ctx.ynode_voltages(), a.ctx.ynode_voltages(), rtol=1e-10)


def test_model_at_state_updates_the_tap_dependent_yprims(prepared: PreparedNetwork) -> None:
    heavy(prepared)
    settle(prepared.ctx, "static_settle", VID)
    assert prepared.ctx.solve().converged
    model = model_at_state(prepared.model, prepared.ctx)
    for before, after in zip(prepared.model.elements, model.elements, strict=True):
        assert before.name == after.name
        if before.yprim is None:
            assert after.yprim is None
        elif before.is_tap_dependent and before.element_class == "transformer":
            assert not np.allclose(after.yprim, before.yprim)  # new tap
        elif not before.is_tap_dependent:
            assert np.array_equal(after.yprim, before.yprim)  # type: ignore[arg-type]


# ------------------------------------------------------------------- registry


def test_states_are_deduplicated(prepared: PreparedNetwork) -> None:
    registry = StateRegistry()
    a = read_state(prepared.ctx, VID, "as_is")
    b = NetworkState(VID, (("reg", (1.0, 1.0125)),), a.capacitors, "static_settle")
    assert registry.add(a) == a.state_id
    assert registry.add(b) == b.state_id
    assert registry.add(a) == a.state_id
    table = registry.table()
    schema.check_schema(table, schema.STATES, "states/states")
    rows = {r["state_id"]: r for r in table.to_pylist()}
    assert rows[a.state_id]["n_scenarios"] == 2
    assert rows[b.state_id]["n_scenarios"] == 1
    assert json.loads(rows[b.state_id]["taps"]) == {"reg": [1.0, 1.0125]}
