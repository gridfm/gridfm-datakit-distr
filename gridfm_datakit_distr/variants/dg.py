"""The capacity of a network's dispatchable generators (DG).

From the DG case study of Geth, Claeys and Deconinck ("Optimal power flow in four-wire distribution
networks: formulation and benchmarking", PSCC 2022, §IV-B): a DG at every 4th load
(:func:`~gridfm_datakit_distr.variants.perturbation.dg_sites`), every DG with the same capacity
``P^max``, "sized such that [...] the DGs can congest the network". The paper gives no number and no
procedure. Ours:

- ``P^max`` is the smallest equal capacity at which the units congest the unperturbed network: at
  its nominal loads, without PV and storage, every unit at ``p = P``, ``q = 0``, controls off
  (the taps as compiled), solved in OpenDSS;
- *congest*: a bounded voltage of the OPF (:func:`~gridfm_datakit_distr.opf.support.vpn_bounds`)
  reaches the OPF's upper bound, or a transformer the OPF bounds reaches its rating at winding 1
  (:func:`~gridfm_datakit_distr.opf.support.winding_va`, the OPF's only branch limit), counting
  only the transformers below their rating at ``P = 0``.

Search. ``r(P)``, the largest of those ratios (value over bound), is close to convex in ``P``: the
voltages rise with ``P``, a transformer's flow falls, then rises. A convex ``r`` below 1 at
``P = 0`` crosses 1 once, so a scan ``P = s · 2^k`` (``k = 0 … SCAN_STEPS - 1``, ``s`` the
customers' total nominal apparent power per unit) brackets the crossing between the last point
below every bound and the first at one, and a bisection narrows the bracket to :data:`BISECT_REL`
of ``P``. A scan step can jump over the crossing to a point where the power flow diverges (EDP
urban): then the step itself is bisected until a converged point at a bound is found, or the
divergence lies within :data:`BISECT_REL` of the last point below every bound. ``P^max`` is the
bracket's upper end, re-solved and checked last. A network without a
capacity gets no DG, and a status that says why (:class:`DGCapacity`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PrepareOptions, prepare_network
from gridfm_datakit_distr.opf.support import vpn_bounds, winding_va
from gridfm_datakit_distr.scenarios.profiles import customers_of
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from gridfm_datakit_distr.variants.perturbation import Perturbation, dg_sites, edit_script
from gridfm_datakit_distr.variants.targets import VariantTargets

SCAN_STEPS = 7
"""Scan points ``s · 2^k``, ``k = 0 … 6``: the largest capacity tried is 64 times the customers'
total nominal apparent power per unit (an implementation limit, not a model parameter)."""
BISECT_REL = 1e-3
"""Width of the final bracket, relative to its upper end."""

Probe = Callable[[float], tuple[bool, float]]
"""Every unit at ``P`` (W) -> (converged, the largest ratio of a limit to its bound)."""


@dataclass(frozen=True, slots=True)
class DGCapacity:
    """The capacity of a network's DG and why, as ``manifest.json`` records it."""

    status: str
    """``ok``; ``no_site`` (fewer than 4 loads); ``voltage_at_bound`` (a voltage at or above the
    upper bound without DG); ``no_binding_limit`` (nothing reached by the last scan point);
    ``not_converged`` (the power flow diverges before a limit is reached, within
    :data:`BISECT_REL`, or a solve of the final bisection did not converge);
    ``off_setpoint`` (at ``P^max`` a unit or a load leaves its set-point)."""
    n_units: int
    p_max_w: float | None = None
    """``P^max`` (W) with ``ok``, else None."""
    binding: str | None = None
    """``voltage``, or the transformer whose rating binds."""
    ignored: tuple[str, ...] = ()
    """Rated transformers above their rating at ``P = 0``, not counted."""

    @property
    def kw(self) -> float | None:
        """``P^max`` in kW, None without a capacity."""
        return None if self.p_max_w is None else self.p_max_w / 1e3

    def record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_units": self.n_units,
            "p_max_w": self.p_max_w,
            "binding": self.binding,
            "ignored": list(self.ignored),
        }


def search(probe: Probe, scale: float) -> tuple[str, float, float]:
    """Status and bracket ``(lo, hi)`` (W) of the first ``P`` at which ``probe``'s ratio reaches 1,
    from ``P = 0`` (below every bound); ``hi`` is NaN unless the status is ``ok``. Only a converged
    point at a bound counts: a scan step that diverges is refined (:func:`_past_divergence`)."""
    lo = 0.0
    for k in range(SCAN_STEPS):
        p = scale * 2.0**k
        converged, ratio = probe(p)
        if not converged:
            return _past_divergence(probe, lo, p)
        if ratio >= 1.0:
            return _bisect(probe, lo, p)
        lo = p
    return "no_binding_limit", lo, math.nan


def _past_divergence(probe: Probe, lo: float, bad: float) -> tuple[str, float, float]:
    """A scan step from ``lo`` (converged, below every bound) to ``bad`` (diverged), which may
    have stepped over a bound: bisect between them until a converged point at a bound is found
    (then :func:`_bisect`) or the gap is within :data:`BISECT_REL` (``not_converged``). A solve
    starts from the previous one, and a diverged one leaves a bad start (spike s54): after each
    divergence the next try is preceded by a solve at ``lo``."""
    diverged = True
    while bad - lo > BISECT_REL * bad:
        if diverged and not probe(lo)[0]:
            return "not_converged", lo, math.nan
        mid = 0.5 * (lo + bad)
        converged, ratio = probe(mid)
        diverged = not converged
        if diverged:
            bad = mid
        elif ratio >= 1.0:
            return _bisect(probe, lo, mid)
        else:
            lo = mid
    return "not_converged", lo, math.nan


def _bisect(probe: Probe, lo: float, hi: float) -> tuple[str, float, float]:
    """Narrow ``(lo, hi)``, below every bound at ``lo`` and at one at ``hi``, to
    :data:`BISECT_REL` of ``hi``."""
    while hi - lo > BISECT_REL * hi:
        mid = 0.5 * (lo + hi)
        converged, ratio = probe(mid)
        if not converged:
            return "not_converged", lo, math.nan
        if ratio >= 1.0:
            hi = mid
        else:
            lo = mid
    return "ok", lo, hi


def dg_capacity(
    network: Network,
    targets: VariantTargets,
    vmax_pu: float,
    options: PrepareOptions | None = None,
    root: Path | None = None,
) -> DGCapacity:
    """The DG capacity of ``network`` (``targets``: its unperturbed targets) under the OPF's upper
    voltage bound ``vmax_pu`` (pu of each pair's nominal voltage)."""
    sites = dg_sites(targets)
    if not sites:
        return DGCapacity("no_site", 0)
    opts = options or PrepareOptions()
    units = Perturbation(0, dg_kw={c.element: 0.0 for c in sites})
    script = edit_script(units, targets, (opts.vminpu, opts.vmaxpu))  # as the variants do
    prepared = prepare_network(network, opts, root, edits=script)
    ctx, model = prepared.ctx, prepared.model
    api = ctx.api
    index = {name.lower(): j for j, name in enumerate(api.Generators.AllNames(), start=1)}
    engine = [index[f"dg_{c.name}".lower()] for c in sites]
    positions = ctx.ynode_positions(n.opendss_node for n in model.nodes)
    pairs, _, v_hi = vpn_bounds(model, 0.0, vmax_pu)
    rated = [
        (e, e.rating_va)
        for e in model.elements
        if e.enabled and e.element_class == "transformer" and e.rating_va
    ]

    def ratios(p_w: float) -> NDArray[np.float64] | None:
        for j in engine:
            api.Generators.Idx(j)
            api.Generators.kW(p_w / 1e3)
            api.Generators.kvar(0.0)
        if not ctx.solve().converged:
            return None
        v = ctx.ynode_voltages()[positions]
        neutral = np.where(pairs[:, 1] == GROUND, 0.0, v[pairs[:, 1]])
        vpn = np.abs(v[pairs[:, 0]] - neutral)
        return np.r_[np.max(vpn / v_hi), [winding_va(e, v) / va for e, va in rated]]

    at_zero = ratios(0.0)
    if at_zero is None:
        return DGCapacity("not_converged", len(sites))
    counted = np.r_[True, at_zero[1:] < 1.0]
    ignored = tuple(e.name for (e, _), over in zip(rated, ~counted[1:], strict=True) if over)
    if at_zero[0] >= 1.0:
        return DGCapacity("voltage_at_bound", len(sites), ignored=ignored)

    def probe(p_w: float) -> tuple[bool, float]:
        r = ratios(p_w)
        return (False, math.nan) if r is None else (True, float(np.max(r[counted])))

    total = sum(abs(complex(c.p_w, c.q_var)) for c in customers_of(prepared))
    status, lo, hi = search(probe, total / len(sites))
    if status != "ok":
        return DGCapacity(status, len(sites), ignored=ignored)
    below, top = ratios(lo), ratios(hi)  # the bracket again, the upper end last
    if below is None or top is None:
        return DGCapacity("not_converged", len(sites), ignored=ignored)
    if not setpoint_report(ctx).passes():
        return DGCapacity("off_setpoint", len(sites), ignored=ignored)
    if not np.max(below[counted]) < 1.0 <= np.max(top[counted]):
        raise RuntimeError(f"{network.id}: the DG capacity's bracket did not reproduce")
    names = ["voltage", *(e.name for e, _ in rated)]
    binding = names[int(np.argmax(np.where(counted, top, -np.inf)))]
    return DGCapacity("ok", len(sites), hi, binding, ignored)
