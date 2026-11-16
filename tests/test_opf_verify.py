"""Verification of an OPF dispatch in OpenDSS on the inline feeder (needs the ``opf`` extra)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.opf.verify import TOLERANCE_PU, verify
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
