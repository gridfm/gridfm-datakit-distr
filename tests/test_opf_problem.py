"""The OPF problem on an inline four-wire feeder (needs the ``opf`` extra)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("casadi")

from gridfm_datakit_distr.engine.context import complex_from_interleaved
from gridfm_datakit_distr.extract.source import Source, read_source
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.opf.problem import (
    SOLVED,
    Costs,
    OPFProblem,
    OPFResult,
    PowerFlowStart,
    _impedance_rows,
    economic,
)
from gridfm_datakit_distr.scenarios.profiles import customers_of
from gridfm_datakit_distr.solve.pf_batch import (
    NEG_SEQUENCE_SOURCE,
    add_negative_sequence_source,
    negative_sequence_current,
)
from tests.circuits import ALL_CLASSES, OPF_FEEDER, write_circuit

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


def test_a_network_outside_the_scope_is_refused(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", ALL_CLASSES)
    prepared = prepare_network(
        Network(id="t/other", source="src", master="master.dss"), root=tmp_path
    )
    with pytest.raises(ValueError, match="pvsystem not covered"):
        OPFProblem(prepared.model, read_source(prepared.ctx, prepared.model), VMIN_PU, VMAX_PU)


def test_different_costs_give_different_dispatches(case: Case) -> None:
    """With an overvoltage to remove, the costs decide which unit curtails."""
    customers = customers_of(case.prepared)

    def dispatch(pv_c1: list[float]) -> np.ndarray:
        result = case.problem.solve(
            source_v_pu=1.0,
            load_p_w=[c.p_w for c in customers],
            load_q_var=[c.q_var for c in customers],
            pv_p_avail_w=[30e3, 30e3],
            pv_s_rated_va=[30e3, 30e3],
            costs=Costs(c1=np.array([1.0, *pv_c1]), c2=np.zeros(3)),
        )
        assert result.status == SOLVED
        return result.pv_p_w

    cheap_h2, cheap_h3 = dispatch([0.1, 0.9]), dispatch([0.9, 0.1])
    assert cheap_h2[0] > cheap_h3[0] + 1.0  # W
    assert cheap_h3[1] > cheap_h2[1] + 1.0


def nlp_rows(
    case: Case, x: np.ndarray, source_v_pu: float = 1.0, u2: float = 0.0, angle: float = 0.0
) -> np.ndarray:
    """The NLP's constraint rows at ``x`` (per unit as the problem scales them)."""
    costs = economic(2)
    par = case.problem.parameters(source_v_pu, u2, angle, costs.c1, costs.c2)
    g = case.problem._solver.get_function("nlp_g")(x=x, p=par)["g"]
    return np.asarray(g, dtype=np.float64).ravel()


def source_rows(case: Case, g: np.ndarray) -> np.ndarray:
    """The KCL rows of the source nodes as complex numbers (real parts, then imaginary parts)."""
    src = case.source.nodes
    return g[src] + 1j * g[case.problem._n + src]


def test_the_source_rows_are_written_in_impedance_form(case: Case) -> None:
    """``E - V - Z_s I = 0`` at the source nodes: with every voltage and current at zero the rows
    are minus the ideal source voltages (order 1 in per unit), not ``Y_s E`` (order |Y_s|)."""
    n_x = case.problem._solver.size1_in("x0")
    g = nlp_rows(case, np.zeros(n_x), source_v_pu=1.02)
    e_pu = 1.02 * case.source.e_v / case.problem._vb[case.source.nodes]
    np.testing.assert_allclose(source_rows(case, g), -e_pu, rtol=0, atol=1e-12)
    src = case.source.nodes
    others = np.delete(g[: 2 * case.problem._n], np.r_[src, case.problem._n + src])
    assert not others.any()


def test_the_source_rows_hold_at_the_power_flow(case: Case) -> None:
    """At the engine's power-flow solution the source rows are satisfied to the solver's
    tolerance: the stiff source admittance no longer amplifies the engine's rounding."""
    ctx = case.prepared.ctx
    assert ctx.solve().converged  # PV units at their compiled 2 kW, 0 kvar
    v = ctx.ynode_voltages()[case.to_model]
    s = case.problem._s_base
    load = np.array([complex(c.p_w, c.q_var) for c in customers_of(case.prepared)]) / s
    x = case.problem._start_from(PowerFlowStart(v, np.full(2, 2e3), np.zeros(2)), load)
    assert np.max(np.abs(source_rows(case, nlp_rows(case, x)))) <= 1e-9


def test_the_source_rows_hold_at_an_unbalanced_power_flow(tmp_path: Path) -> None:
    """With a negative sequence realised in the engine (its current source), the source rows of
    the same ratio and angle hold at the engine's solution; the balanced rows, or the angle
    reversed, do not: the OPF's ``E`` has the engine's sign and phase order."""
    case = make_case(tmp_path)
    ctx = case.prepared.ctx
    source = add_negative_sequence_source(ctx, case.prepared.model)
    u2, angle = 0.02, 2.2
    i_neg = negative_sequence_current(source, 1.0, u2, angle)
    ctx.api.Isource.Name(NEG_SEQUENCE_SOURCE)
    ctx.api.Isource.Amps(abs(i_neg))
    ctx.api.Isource.AngleDeg(float(np.degrees(np.angle(i_neg))))
    assert ctx.solve().converged
    v = ctx.ynode_voltages()[case.to_model]
    s = case.problem._s_base
    load = np.array([complex(c.p_w, c.q_var) for c in customers_of(case.prepared)]) / s
    x = case.problem._start_from(PowerFlowStart(v, np.full(2, 2e3), np.zeros(2)), load)
    assert np.max(np.abs(source_rows(case, nlp_rows(case, x, 1.0, u2, angle)))) <= 1e-9
    for wrong in ((0.0, 0.0), (u2, -angle)):
        assert np.max(np.abs(source_rows(case, nlp_rows(case, x, 1.0, *wrong)))) > 1e-3


def test_impedance_rows_change_only_the_source_rows() -> None:
    """The source rows become ``Z_s`` times the complex rows; every other row is unchanged."""
    rng = np.random.default_rng(0)
    n, m, src = 5, 7, np.array([1, 3])
    a = rng.normal(size=(2 * n, m))
    b = rng.normal(size=2 * n)
    ys = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) + 3 * np.eye(2)
    a2, b2 = _impedance_rows(sp.csr_matrix(a), b, src, ys, n)
    a2 = a2.toarray()
    rows = np.r_[src, n + src]
    keep = np.setdiff1d(np.arange(2 * n), rows)
    np.testing.assert_array_equal(a2[keep], a[keep])
    np.testing.assert_array_equal(b2[keep], b[keep])
    z = np.linalg.inv(ys)
    np.testing.assert_allclose(a2[src] + 1j * a2[n + src], z @ (a[src] + 1j * a[n + src]))
    np.testing.assert_allclose(b2[src] + 1j * b2[n + src], z @ (b[src] + 1j * b[n + src]))
