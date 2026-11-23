"""The OPF's device models against OpenDSS: transformers, three-wire buses, wye, delta and
multi-phase loads and PV units; the transformer rating; the scaling on a medium-voltage feeder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.extract.source import read_source
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.opf.problem import (
    SOLVED,
    OPFProblem,
    OPFResult,
    PowerFlowStart,
    economic,
    power_base,
)
from gridfm_datakit_distr.opf.support import unsupported_reason, winding_va
from gridfm_datakit_distr.opf.verify import TOLERANCE_PU, verify, voltage_gap
from gridfm_datakit_distr.scenarios.profiles import customers_of
from tests.circuits import DELTA_DELTA, LATERAL, OPF_DEVICES, write_circuit

WIDE = (0.5, 1.5)
"""Voltage limits that never bind here: the solution is the power flow at the dispatch."""


@dataclass
class Case:
    prepared: PreparedNetwork
    problem: OPFProblem


def build(root: Path, circuit: dict[str, str]) -> Case:
    write_circuit(root / "src", circuit)
    prepared = prepare_network(Network(id="t/dev", source="src", master="master.dss"), root=root)
    customers = customers_of(prepared)
    s_base = power_base([abs(complex(c.p_w, c.q_var)) for c in customers])
    source = read_source(prepared.ctx, prepared.model)
    return Case(prepared, OPFProblem(prepared.model, source, *WIDE, 300.0, s_base))


@pytest.fixture(scope="module")
def devices(tmp_path_factory: pytest.TempPathFactory) -> Case:
    return build(tmp_path_factory.mktemp("opf_devices"), OPF_DEVICES)


def solve(case: Case, p_avail_w: list[float], q_max_var: list[float] | None = None) -> OPFResult:
    customers = customers_of(case.prepared)
    assert tuple(c.element for c in customers) == case.problem.load_names
    n = len(case.problem.pv_names)
    api = case.prepared.ctx.api
    api.Vsources.First()
    return case.problem.solve(
        source_v_pu=float(api.Vsources.PU()),  # the circuit's own source voltage
        load_p_w=[c.p_w for c in customers],
        load_q_var=[c.q_var for c in customers],
        pv_p_avail_w=p_avail_w,
        pv_s_rated_va=[1e6] * n,
        costs=economic(n),
        pv_q_max_var=[0.0] * n if q_max_var is None else q_max_var,
    )


def test_every_device_is_covered(devices: Case) -> None:
    assert unsupported_reason(devices.prepared.model) is None
    assert devices.problem.pv_names == (
        "Generator.pv_delta3",
        "Generator.pv_house",
        "Generator.pv_wye3",
    )
    assert devices.problem.transformer_names == ("Transformer.t",)
    model = devices.prepared.model
    three_wire = {n.bus for n in model.nodes if n.bus.startswith("mv")}
    assert not any(n.conductor == "n" for n in model.nodes if n.bus in three_wire)


def test_a_fixed_dispatch_reproduces_the_opendss_power_flow(devices: Case) -> None:
    """With no limit binding, the OPF keeps every unit at its available power (q held at 0):
    its solution is the power flow OpenDSS solves at the same set points."""
    p_avail = [10e3, 3e3, 10e3]  # the units' compiled outputs
    result = solve(devices, p_avail)
    assert result.status == SOLVED
    np.testing.assert_allclose(result.pv_p_w, p_avail, rtol=1e-6)
    np.testing.assert_allclose(result.pv_q_var, 0.0, atol=1e-3)
    check = verify(devices.prepared.ctx, devices.prepared.model, devices.problem, result)
    assert check.verified, check.reason
    assert check.max_dv_pu <= TOLERANCE_PU


def test_reverse_flow_is_curtailed_to_the_transformer_rating(devices: Case) -> None:
    p_avail = [250e3, 3e3, 250e3]  # far more than the loads and the 250 kVA transformer
    result = solve(devices, p_avail, q_max_var=[0.0, 0.0, 0.0])
    assert result.status == SOLVED
    assert sum(p_avail) - result.pv_p_w.sum() > 1e3
    (xfm,) = [e for e in devices.prepared.model.elements if e.name == "Transformer.t"]
    assert winding_va(xfm, result.v) == pytest.approx(250e3, rel=1e-6)
    check = verify(devices.prepared.ctx, devices.prepared.model, devices.problem, result)
    assert check.verified, check.reason


def test_the_power_base_follows_the_loads() -> None:
    assert power_base([800.0, 1200.0, 3000.0]) == 1e3  # 5 kVA in all
    assert power_base([2e5, 5e5, 9e5]) == 1e5  # 1.6 MVA in all
    assert power_base([]) == 1e3
    assert power_base([10.0]) == 1e3  # never below 1 kVA


def test_a_medium_voltage_feeder_solves_on_its_power_base(tmp_path: Path) -> None:
    case = build(tmp_path, LATERAL)  # 11 kV, loads of 104 and 538 kVA
    assert case.problem._s_base == 1e4  # a tenth of 642 kVA, floored
    result = solve(case, [])
    assert result.status == SOLVED
    check = verify(case.prepared.ctx, case.prepared.model, case.problem, result)
    assert check.verified, check.reason


def test_a_start_at_the_power_flow_reaches_the_same_optimum(devices: Case) -> None:
    """The second start of the labeller: at the power flow of the same scenario."""
    ctx = devices.prepared.ctx
    p_avail = [10e3, 3e3, 10e3]  # the units' compiled outputs: the power flow ctx holds
    order = {name: j for j, name in enumerate(ctx.ynode_order())}
    v = ctx.ynode_voltages()[[order[n.opendss_node.lower()] for n in devices.prepared.model.nodes]]
    start = PowerFlowStart(v=v, pv_p_w=np.array(p_avail), pv_q_var=np.zeros(3))
    flat = solve(devices, p_avail)
    customers = customers_of(devices.prepared)
    ctx.api.Vsources.First()
    warm = devices.problem.solve(
        source_v_pu=float(ctx.api.Vsources.PU()),
        load_p_w=[c.p_w for c in customers],
        load_q_var=[c.q_var for c in customers],
        pv_p_avail_w=p_avail,
        pv_s_rated_va=[1e6] * 3,
        costs=economic(3),
        pv_q_max_var=[0.0] * 3,
        start=start,
    )
    assert (flat.status, warm.status) == (SOLVED, SOLVED)
    assert warm.objective == pytest.approx(flat.objective, rel=1e-9)
    assert warm.iterations < flat.iterations


def test_a_zone_grounded_by_convention_is_bounded_and_verified_phase_to_phase(
    tmp_path: Path,
) -> None:
    """Bus x floats behind a delta-delta transformer: its bounds are phase to phase, on √3 times
    the nominal line-to-neutral voltage, and the dispatch verifies in OpenDSS."""
    case = build(tmp_path, DELTA_DELTA)
    model = case.prepared.model
    x = sorted(n.idx for n in model.nodes if n.bus == "x")
    assert [list(z) for z in case.problem.conventional] == [x]
    pairs = [tuple(p) for p in case.problem.pairs if p[0] in x]
    assert pairs == [(x[0], x[1]), (x[1], x[2]), (x[2], x[0])]
    at_x = np.isin(case.problem.pairs[:, 0], x)
    nominal = np.sqrt(3.0) * model.nodes[x[0]].nominal_v_ln
    np.testing.assert_allclose(case.problem.vpn_hi_v[at_x], WIDE[1] * nominal, rtol=1e-12)
    result = solve(case, [20e3])
    assert result.status == SOLVED
    check = verify(case.prepared.ctx, model, case.problem, result)
    assert check.verified, check.reason


def test_the_gap_ignores_a_common_mode_shift_only() -> None:
    """In a zone grounded by convention a common-mode shift passes and a perturbed
    phase-to-phase voltage fails: the comparison is not vacuous."""
    base = np.full(5, 1000.0)
    v = np.exp(-2j * np.pi / 3 * np.arange(5)) * 1000.0
    zone = [np.array([2, 3, 4], dtype=np.int64)]
    shifted = v.copy()
    shifted[2:] += 37.0 - 12.0j  # the zone's common mode moves
    assert voltage_gap(shifted, v, base, zone) <= 1e-12
    assert voltage_gap(shifted, v, base, []) > 1e-2  # without the zone it would fail
    perturbed = shifted.copy()
    perturbed[3] += 2e-3  # one phase-to-phase voltage moves by 2e-6 pu
    assert voltage_gap(perturbed, v, base, zone) > TOLERANCE_PU
    outside = v.copy()
    outside[0] += 2e-3
    assert voltage_gap(outside, v, base, zone) > TOLERANCE_PU
