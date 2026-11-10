"""Tests for the vectorised result extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.scenarios.controls import model_at_state, settle
from gridfm_datakit_distr.solve.extraction import build_plan
from tests.circuits import CONTROLLED, VARIANT_FEEDER, write_circuit


def engine_currents(ctx: DssContext, name: str) -> np.ndarray:
    ctx.activate_element(name)
    raw = np.asarray(ctx.api.CktElement.Currents())
    return raw[0::2] + 1j * raw[1::2]


def check_against_engine(ctx: DssContext, model: ElectricalModel) -> None:
    v = ctx.ynode_voltages()
    order = {n: j for j, n in enumerate(ctx.ynode_order())}
    v_model = v[[order[n.opendss_node.lower()] for n in model.nodes]]
    plan = build_plan(model)
    i_local = plan.currents(v_model)
    for e in model.pd_elements():
        rows = plan.element_idx == e.idx
        expected = engine_currents(ctx, e.name)
        terminals = [t for t, term in enumerate(e.terminals) for _ in term]
        assert plan.terminal[rows].tolist() == terminals
        scale = np.abs(expected).max()
        np.testing.assert_allclose(i_local[rows], expected, rtol=0, atol=1e-8 * scale)
    i_inj = plan.injections(i_local)
    np.testing.assert_allclose(i_inj, model.ybus() @ v_model, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("circuit", [CONTROLLED, VARIANT_FEEDER], ids=["controlled", "variant"])
def test_currents_match_the_engine(tmp_path: Path, circuit: dict[str, str]) -> None:
    write_circuit(tmp_path / "src", circuit)
    prepared = prepare_network(Network(id="t/x", source="src", master="master.dss"), root=tmp_path)
    check_against_engine(prepared.ctx, prepared.model)


def test_tap_dependent_blocks_follow_the_state(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", CONTROLLED)
    prepared = prepare_network(Network(id="t/x", source="src", master="master.dss"), root=tmp_path)
    plan = build_plan(prepared.model)
    prepared.ctx.command("Edit Load.ld kW=900 kvar=300")
    settle(prepared.ctx, "static_settle", "t/x/v0000")
    assert prepared.ctx.solve().converged
    at_state = model_at_state(prepared.model, prepared.ctx)
    check_against_engine(prepared.ctx, at_state)  # the state's Yprims reproduce the engine
    moved = plan.at_state(at_state)
    assert (moved.fixed != plan.fixed).nnz == 0
    assert (moved.dependent != plan.dependent).nnz > 0  # the regulator's new tap
    fresh = build_plan(at_state)
    assert abs(moved.yblk - fresh.yblk).max() == 0
