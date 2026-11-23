"""Verification of an OPF dispatch in OpenDSS.

The dispatch is set on the PV units and DG of a context that holds the scenario (loads, storage
set-points and source voltage set, controls frozen), the power flow is solved with the controls off
at the context's label tolerance, and the solution is checked: every load and generator at its set
point (:func:`setpoint_report`), node voltages equal to the OPF's within :data:`TOLERANCE_PU`
of each node's base (in a zone whose voltage to ground is a numerical convention, the voltages
relative to the zone's first node: its common mode is no label), the bounded voltages
within the OPF's bounds up to :data:`TOLERANCE_PU`. The NLP
and OpenDSS share the admittances, so the two solutions agree to the solvers' accuracy. The
generators get back their outputs afterwards; a failed check is returned, never raised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel
from gridfm_datakit_distr.opf.problem import OPFProblem, OPFResult
from gridfm_datakit_distr.opf.support import pair_nominal_v
from gridfm_datakit_distr.scenarios.controls import (
    MAX_CONTROL_ITER,
    apply_state,
    has_controls,
    read_state,
)
from gridfm_datakit_distr.validate.setpoints import setpoint_report

TOLERANCE_PU = 1e-6
"""Largest voltage difference and bound excess accepted, per unit of the node's base."""


@dataclass(frozen=True, slots=True)
class Verification:
    """The outcome of one verification."""

    verified: bool
    reason: str | None
    """The failed checks, ``None`` when verified."""
    converged: bool
    max_dv_pu: float
    """Largest ``|V_opendss - V_opf|`` per unit of the node's base; NaN without a power flow."""
    max_bound_excess_pu: float
    """Largest excess of ``|V_p - V_n|`` over its bounds, per unit of the phase node's nominal
    voltage; 0 within the bounds, NaN without a power flow."""
    controls_hold: bool | None = None
    """With regulators or switched capacitors: whether, at the solved dispatch, a solve with the
    controls on (``ControlMode=STATIC``) keeps the scenario's taps and steps; None without
    controls. It does not enter ``verified``: the label is the optimum at the frozen state."""


def verify(
    ctx: DssContext, model: ElectricalModel, problem: OPFProblem, result: OPFResult
) -> Verification:
    """Check ``result`` (a solve of ``problem`` for the scenario ``ctx`` holds) in OpenDSS."""
    api = ctx.api
    before: list[tuple[float, float]] = []
    names = problem.pv_names + problem.dg_names
    p_set, q_set = np.r_[result.pv_p_w, result.dg_p_w], np.r_[result.pv_q_var, result.dg_q_var]
    try:
        for name, p_w, q_var in zip(names, p_set, q_set, strict=True):
            api.Generators.Name(name.split(".", 1)[1])
            before.append((float(api.Generators.kW()), float(api.Generators.kvar())))
            api.Generators.kW(float(p_w) / 1e3)
            api.Generators.kvar(float(q_var) / 1e3)
        ctx.command("Set ControlMode=OFF")
        ctx.command("Set Mode=Snap")
        status = ctx.solve()
        if not status.converged:
            return Verification(False, "power flow not converged", False, np.nan, np.nan, None)
        report = setpoint_report(ctx)
        v = ctx.ynode_voltages()[ctx.ynode_positions(n.opendss_node for n in model.nodes)]
        hold = _controls_hold(ctx, model.variant_id) if has_controls(ctx) else None
    finally:
        for name, (kw, kvar) in zip(names, before, strict=False):
            api.Generators.Name(name.split(".", 1)[1])
            api.Generators.kW(kw)
            api.Generators.kvar(kvar)
    base_v = np.array([n.kv_base_ln for n in model.nodes]) * 1e3
    dv = voltage_gap(v, result.v, base_v, problem.conventional)
    pairs = problem.pairs
    neutral = np.where(pairs[:, 1] == GROUND, 0.0, v[pairs[:, 1]])
    vpn = np.abs(v[pairs[:, 0]] - neutral)
    nominal = pair_nominal_v(model, pairs)
    excess = np.maximum(vpn - problem.vpn_hi_v, problem.vpn_lo_v - vpn) / nominal
    excess_max = max(float(np.max(excess, initial=0.0)), 0.0)
    failed = []
    if not report.passes():
        failed.append(f"set points: {len(report.off)} off, worst {report.worst}")
    if not dv <= TOLERANCE_PU:
        failed.append(f"voltages differ from the OPF's by {dv:.2e} pu")
    if not excess_max <= TOLERANCE_PU:
        failed.append(f"voltage bounds exceeded by {excess_max:.2e} pu")
    reason = "; ".join(failed) or None
    return Verification(reason is None, reason, True, dv, excess_max, hold)


def voltage_gap(
    v: NDArray[np.complex128],
    v_opf: NDArray[np.complex128],
    base_v: NDArray[np.float64],
    zones: Sequence[NDArray[np.int64]],
) -> float:
    """Largest ``|V - V_opf|`` per unit of each node's base; inside each of ``zones`` (voltage to
    ground a numerical convention) the differences relative to the zone's first node, so that a
    common-mode shift of the zone alone passes and any change of a voltage within it does not."""
    gap = v - v_opf
    for zone in zones:
        gap[zone] = gap[zone] - gap[zone[0]]
    return float(np.max(np.abs(gap) / base_v))


def _controls_hold(ctx: DssContext, variant_id: str) -> bool:
    """Whether the controls keep the frozen state at the solved dispatch: one more solve with
    ``ControlMode=STATIC`` from it, then the frozen state and ``ControlMode=OFF`` back."""
    frozen = read_state(ctx, variant_id, "as_is")
    try:
        ctx.command(f"Set ControlMode=STATIC MaxControlIter={MAX_CONTROL_ITER}")
        ctx.command("Set Mode=Snap")
        status = ctx.solve()
        moved = read_state(ctx, variant_id, "as_is")
    finally:
        apply_state(ctx, frozen)
        ctx.command("Set ControlMode=OFF")
    return status.converged and moved.state_id == frozen.state_id
