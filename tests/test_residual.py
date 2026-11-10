"""Tests for the KCL residual, the C3 admittance cross-check and residual sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.extract.injections import injections, pc_source_admittance
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.validate.consistency import compare_admittance
from gridfm_datakit_distr.validate.residual import check_this_scenario, residual
from tests.circuits import CONTROLLED, FOUR_WIRE, LATERAL, TWO_BUS, write_circuit

CIRCUITS = {
    "two_bus": TWO_BUS,
    "lateral": LATERAL,
    "four_wire": FOUR_WIRE,
    "controlled": CONTROLLED,
}


def solved(tmp_path: Path, files: dict[str, str]) -> tuple[DssContext, ElectricalModel]:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files))
    ctx.set_options(tolerance=1e-10, max_iterations=100)
    assert ctx.solve().converged
    return ctx, extract_electrical(ctx, "test", "test/base")


@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_kcl_closes_on_inline_circuits(tmp_path: Path, circuit: str) -> None:
    ctx, model = solved(tmp_path, CIRCUITS[circuit])
    v = ctx.ynode_voltages()
    i_pc = injections(ctx, model)
    report = residual(model, v, i_pc, tolerance=1e-10)
    assert report.passes(), report
    assert report.rel <= 1e-8, report  # the plan's target
    assert report.backward < 1e-10, report
    assert report.n_checked == model.n_nodes - model.source_nodes().size
    # The sources balance the source nodes: with them KCL closes everywhere.
    full = model.ybus() @ v - i_pc - injections(ctx, model, sources=True)
    assert np.abs(full).max() <= 1e-8 * max(1.0, np.abs(i_pc).max())


@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_wrong_current_sign_is_detected(tmp_path: Path, circuit: str) -> None:
    ctx, model = solved(tmp_path, CIRCUITS[circuit])
    report = residual(model, ctx.ynode_voltages(), -injections(ctx, model), tolerance=1e-10)
    assert not report.passes()


@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_solver_y_is_reproduced(tmp_path: Path, circuit: str) -> None:
    ctx, model = solved(tmp_path, CIRCUITS[circuit])
    report = compare_admittance(
        model, model.ybus(), ctx.system_y(), pc_source_admittance(ctx, model)
    )
    assert report.passes(1e-12), report


def test_admittance_mismatch_is_located(tmp_path: Path) -> None:
    ctx, model = solved(tmp_path, TWO_BUS)
    y_state = model.ybus().tolil()
    y_state[4, 4] *= 1.001  # node load.2
    report = compare_admittance(
        model, y_state.tocsr(), ctx.system_y(), pc_source_admittance(ctx, model)
    )
    assert not report.passes()
    assert report.worst == ("load.2", "load.2")


def test_admittance_is_judged_at_the_scale_of_cancelled_terms(tmp_path: Path) -> None:
    _, model = solved(tmp_path, TWO_BUS)
    n = model.n_nodes

    def single(value: float) -> sp.csr_matrix:
        return sp.csr_matrix(([value], ([0], [1])), shape=(n, n))

    y_state = sp.csr_matrix((n, n))
    y_pc = single(100.0)  # a load alone couples nodes 0 and 1
    noise = compare_admittance(model, y_state, y_pc + single(1e-13), y_pc)
    assert noise.passes()  # round-off of Y_solver - Y_pc, absent from Y_state
    missing = compare_admittance(model, y_state, y_pc + single(5.0), y_pc)
    assert not missing.passes()  # a 5 S coupling the extraction would have lost
    assert missing.max_rel == pytest.approx(5.0 / 105.0)


def test_residual_locates_a_wrong_injection(tmp_path: Path) -> None:
    ctx, model = solved(tmp_path, LATERAL)
    i_pc = injections(ctx, model)
    k = next(n.idx for n in model.nodes if n.opendss_node == "n2.2")
    i_pc[k] += 1.0  # 1 A error
    report = residual(model, ctx.ynode_voltages(), i_pc, tolerance=1e-10)
    assert not report.passes()
    assert report.worst_node == "n2.2"
    assert report.max_abs_a == pytest.approx(1.0, rel=1e-6)


def test_residual_sampling_is_deterministic_and_has_the_right_rate() -> None:
    ids = range(200_000)
    picks = [check_this_scenario(i, 0.01) for i in ids]
    assert picks == [check_this_scenario(i, 0.01) for i in ids]
    assert abs(sum(picks) / len(picks) - 0.01) < 0.002
    assert not any(check_this_scenario(i, 0.0) for i in range(1000))
    assert all(check_this_scenario(i, 1.0) for i in range(1000))
    big = (3 << 47) | (12 << 31) | 7  # scenario ids use the high bits
    assert check_this_scenario(big, 0.5) == check_this_scenario(big, 0.5)


def test_c2_bound_allows_what_the_solver_tolerance_cannot_resolve(tmp_path: Path) -> None:
    """A residual below tol * sum|Y||V| passes even if it is large against the load currents."""
    ctx, model = solved(tmp_path, TWO_BUS)
    v = ctx.ynode_voltages()
    i_pc = injections(ctx, model)
    y = model.ybus()
    g = abs(y) @ np.abs(v)
    k = next(n.idx for n in model.nodes if n.opendss_node == "load.1")
    i_pc[k] += 0.5 * 1e-6 * g[k]  # half of what tol = 1e-6 allows at this node
    assert residual(model, v, i_pc, tolerance=1e-6).passes()
    assert not residual(model, v, i_pc, tolerance=1e-12).passes()
