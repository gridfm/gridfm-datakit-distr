"""Verification of an OPF dispatch in OpenDSS on the inline feeder (needs the ``opf`` extra)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("casadi")

from pathlib import Path

from gridfm_datakit_distr.extract.source import read_source
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.opf.problem import OPFProblem, economic
from gridfm_datakit_distr.opf.verify import TOLERANCE_PU, Verification, verify
from gridfm_datakit_distr.scenarios.controls import model_at_state, read_state, settle
from gridfm_datakit_distr.scenarios.profiles import customers_of
from tests.circuits import CONTROLLED, write_circuit
from tests.test_opf_problem import Case, make_case, solve


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    return make_case(tmp_path_factory.mktemp("opf_verify"))


def uncontrolled(case: Case, pv_kw: float) -> np.ndarray:
    """Set the scenario's PV outputs (unity power factor) and solve its power flow: the state
    the scenario loop leaves the context in. Returns the node voltages (V), model order."""
    ctx = case.prepared.ctx
    for name in case.problem.pv_names:
        ctx.command(f"Edit {name} kW={pv_kw!r} kvar=0")
    ctx.command("Set Mode=Snap")
    assert ctx.solve().converged
    return ctx.ynode_voltages()[case.to_model]


def outputs(case: Case) -> list[tuple[float, float]]:
    api = case.prepared.ctx.api
    out = []
    for name in case.problem.pv_names:
        api.Generators.Name(name.split(".", 1)[1])
        out.append((float(api.Generators.kW()), float(api.Generators.kvar())))
    return out


@pytest.mark.parametrize(("pv_kw", "kva_per_kwp"), [(2.0, 1.25), (30.0, 1.0)])
def test_the_opf_dispatch_verifies(case: Case, pv_kw: float, kva_per_kwp: float) -> None:
    uncontrolled(case, pv_kw)
    result = solve(case, pv_kw=pv_kw, kva_per_kwp=kva_per_kwp)
    assert result.solved
    check = verify(case.prepared.ctx, case.prepared.model, case.problem, result)
    assert check.verified, check.reason
    assert check.reason is None
    assert check.converged
    assert check.max_dv_pu <= TOLERANCE_PU
    assert check.max_bound_excess_pu <= TOLERANCE_PU


def test_a_dispatch_altered_by_hand_fails_on_the_voltages(case: Case) -> None:
    uncontrolled(case, 30.0)
    result = solve(case, pv_kw=30.0, kva_per_kwp=1.0)
    p = result.pv_p_w.copy()
    p[0] *= 1.01
    check = verify(case.prepared.ctx, case.prepared.model, case.problem, replace(result, pv_p_w=p))
    assert not check.verified
    assert check.converged
    assert check.reason is not None
    assert check.reason.startswith("voltages")
    assert check.max_dv_pu > TOLERANCE_PU


def test_the_uncontrolled_power_flow_is_reproduced_afterwards(case: Case) -> None:
    before = uncontrolled(case, 30.0)
    set_before = outputs(case)
    result = solve(case, pv_kw=30.0, kva_per_kwp=1.0)
    assert verify(case.prepared.ctx, case.prepared.model, case.problem, result).verified
    assert outputs(case) == set_before
    ctx = case.prepared.ctx
    ctx.command("Set Mode=Snap")
    assert ctx.solve().converged
    np.testing.assert_array_equal(ctx.ynode_voltages()[case.to_model], before)


def test_controls_hold_tells_whether_the_controls_keep_the_state(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", CONTROLLED)
    prepared = prepare_network(
        Network(id="t/ctrl", source="src", master="master.dss"), root=tmp_path
    )
    ctx = prepared.ctx
    settle(ctx, "static_settle", "t/ctrl/v0000")  # taps and steps the controls hold, frozen
    settled = read_state(ctx, "t/ctrl/v0000", "as_is")
    held = controlled_case_at(prepared)
    assert held.verified, held.reason
    assert held.controls_hold is True
    assert read_state(ctx, "t/ctrl/v0000", "as_is").state_id == settled.state_id  # restored
    (t, taps) = settled.taps[0]
    ctx.api.Transformers.Name(t)
    ctx.api.Transformers.Wdg(2)
    ctx.api.Transformers.Tap(taps[1] + 0.05)  # a state the regulator would leave
    moved = controlled_case_at(prepared)
    assert moved.verified, moved.reason
    assert moved.controls_hold is False


def controlled_case_at(prepared: PreparedNetwork) -> Verification:
    """Verification of the OPF at the state the context holds."""
    ctx = prepared.ctx
    ctx.command("Set ControlMode=OFF")
    ctx.command("Set Mode=Snap")
    assert ctx.solve().converged
    model = model_at_state(prepared.model, ctx)
    problem = OPFProblem(model, read_source(ctx, model), 0.5, 1.5)
    customers = customers_of(prepared)
    ctx.api.Vsources.First()
    result = problem.solve(
        source_v_pu=float(ctx.api.Vsources.PU()),
        load_p_w=[c.p_w for c in customers],
        load_q_var=[c.q_var for c in customers],
        pv_p_avail_w=[],
        pv_s_rated_va=[],
        costs=economic(0),
    )
    assert result.solved
    return verify(ctx, model, problem, result)


def test_without_controls_controls_hold_is_none(case: Case) -> None:
    uncontrolled(case, 2.0)
    result = solve(case, pv_kw=2.0, kva_per_kwp=1.25)
    check = verify(case.prepared.ctx, case.prepared.model, case.problem, result)
    assert check.controls_hold is None
