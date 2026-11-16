"""The OPF problem on an inline four-wire feeder (needs the ``opf`` extra)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.engine.context import complex_from_interleaved
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.opf.problem import (
    SOLVED,
    OPFProblem,
    OPFResult,
    Source,
    economic,
    read_source,
)
from gridfm_datakit_distr.scenarios.profiles import customers_of
from tests.circuits import MV_LV_RNEUT, OPF_FEEDER, write_circuit

VMIN_PU, VMAX_PU = 0.94, 1.10
"""Per unit of the nominal voltage: the feeder's 0.4 kV base, 230.94 V phase-to-neutral."""
REL_TOL = 1e-6
"""Solver accuracy asked of a label: Ipopt runs at tol 1e-9 on per-unit quantities."""


@dataclass
class Case:
    prepared: PreparedNetwork
    source: Source
    problem: OPFProblem
    to_model: np.ndarray


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> Case:
    return make_case(tmp_path_factory.mktemp("opf"))


def make_case(root: Path) -> Case:
    """The inline feeder prepared, its source and its OPF problem."""
    write_circuit(root / "src", OPF_FEEDER)
    prepared = prepare_network(Network(id="t/opf", source="src", master="master.dss"), root=root)
    source = read_source(prepared.ctx, prepared.model)
    problem = OPFProblem(prepared.model, source, VMIN_PU, VMAX_PU)
    order = {name: j for j, name in enumerate(prepared.ctx.ynode_order())}
    to_model = np.array([order[n.opendss_node.lower()] for n in prepared.model.nodes])
    return Case(prepared, source, problem, to_model)


def solve(case: Case, pv_kw: float, kva_per_kwp: float, source_v_pu: float = 1.0) -> OPFResult:
    customers = customers_of(case.prepared)
    assert tuple(c.element for c in customers) == case.problem.load_names
    assert case.problem.pv_names == ("Generator.pv_h2", "Generator.pv_h3")
    return case.problem.solve(
        source_v_pu=source_v_pu,
        load_p_w=[c.p_w for c in customers],
        load_q_var=[c.q_var for c in customers],
        pv_p_avail_w=[pv_kw * 1e3] * 2,
        pv_s_rated_va=[pv_kw * kva_per_kwp * 1e3] * 2,
        costs=economic(2),
    )


def vpn(case: Case, v: np.ndarray) -> np.ndarray:
    nodes = case.prepared.model.nodes
    neutral = {n.bus: n.idx for n in nodes if n.conductor == "n"}
    return np.array([abs(v[n.idx] - v[neutral[n.bus]]) for n in nodes if n.conductor != "n"])


def test_the_source_reproduces_the_power_flow(case: Case) -> None:
    ctx = case.prepared.ctx
    assert case.prepared.solve.converged
    v = ctx.ynode_voltages()[case.to_model]
    ctx.activate_element("Vsource.source")
    i_into = complex_from_interleaved(ctx.api.CktElement.Currents())[:3]
    e = v[case.source.nodes] - np.linalg.solve(case.source.y_s, i_into)
    assert np.max(np.abs(e - case.source.e_v)) < 1e-9 * np.max(np.abs(case.source.e_v))


def test_without_violation_every_unit_keeps_its_available_power(case: Case) -> None:
    result = solve(case, pv_kw=2.0, kva_per_kwp=1.25)
    assert result.status == SOLVED
    assert result.solved
    np.testing.assert_allclose(result.pv_p_w, 2e3, rtol=REL_TOL, atol=0)
    lo, hi = case.problem.vpn_lo_v.min(), case.problem.vpn_hi_v.max()
    assert lo < vpn(case, result.v).min() <= vpn(case, result.v).max() < hi
    # The dispatch, reactive power included, gives OpenDSS the same voltages.
    ctx = case.prepared.ctx
    for name, p, q in zip(case.problem.pv_names, result.pv_p_w, result.pv_q_var, strict=True):
        ctx.command(f"Edit {name} kW={float(p) / 1e3!r} kvar={float(q) / 1e3!r}")
    try:
        assert ctx.solve().converged
        v_dss = ctx.ynode_voltages()[case.to_model]
    finally:
        for name in case.problem.pv_names:
            ctx.command(f"Edit {name} kW=2 kvar=0")
    assert np.max(np.abs(v_dss - result.v)) < 1e-6 * 230.0


def test_overvoltage_is_curtailed_to_the_bound(case: Case) -> None:
    # Inverters rated at their kWp: no reactive power without curtailing (with headroom the
    # reactive power alone clears this overvoltage).
    result = solve(case, pv_kw=30.0, kva_per_kwp=1.0)
    assert result.status == SOLVED
    assert np.sum(30e3 - result.pv_p_w) > 1.0  # W, far above the solver's accuracy
    v_max = vpn(case, result.v).max()
    v_bound = VMAX_PU * 400 / 3**0.5  # V
    assert np.allclose(case.problem.vpn_hi_v, v_bound, rtol=1e-12)
    assert abs(v_max - v_bound) <= REL_TOL * v_bound
    s2 = result.pv_q_var**2 + result.pv_p_w**2
    np.testing.assert_array_less(s2, (30e3) ** 2 * (1 + REL_TOL))


def test_a_model_serves_scenario_after_scenario(case: Case) -> None:
    first = solve(case, pv_kw=2.0, kva_per_kwp=1.25)
    solve(case, pv_kw=30.0, kva_per_kwp=1.0)
    again = solve(case, pv_kw=2.0, kva_per_kwp=1.25)
    assert (again.status, again.objective, again.iterations) == (
        first.status,
        first.objective,
        first.iterations,
    )
    np.testing.assert_array_equal(again.v, first.v)


def test_an_infeasible_scenario_returns_its_status(case: Case) -> None:
    result = solve(
        case, pv_kw=2.0, kva_per_kwp=1.25, source_v_pu=1.2
    )  # 277 V at the source bus itself
    assert not result.solved
    assert result.status != SOLVED
    assert result.status


def test_a_network_outside_the_slice_is_refused(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", MV_LV_RNEUT)
    prepared = prepare_network(
        Network(id="t/mvlv", source="src", master="master.dss"), root=tmp_path
    )
    with pytest.raises(ValueError, match="transformer"):
        OPFProblem(prepared.model, read_source(prepared.ctx, prepared.model), VMIN_PU, VMAX_PU)
