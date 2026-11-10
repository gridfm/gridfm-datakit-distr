"""Setpoint check of a solved operating point.

OpenDSS changes the model of a load or generator outside its voltage window and still reports
convergence. A solution is accepted only if every enabled load and
generator delivers its setpoint:

    |S_delivered - S_set| <= rel_tol * max(|S_set|, 1 VA),   rel_tol = 1e-6,

with ``S_delivered`` the sum of ``CktElement.Powers()`` over the element's conductors (consumed
power; a generator's setpoint is minus its kW + j kvar). PV systems and storage units have no
single setpoint and are not checked.

The delivered power lags the setpoint by about the last voltage update of the solver: measured
0.1 to 1.7 times ``Tolerance``, at most about 3.4 times by a first-order
bound. The check therefore requires ``Tolerance <= rel_tol / 100``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext, complex_from_interleaved

TOLERANCE_MARGIN = 100.0
"""``rel_tol`` must exceed the solver tolerance by this factor."""


@dataclass(frozen=True, slots=True)
class SetpointReport:
    """Delivered power of loads and generators against their setpoints."""

    n_checked: int
    off: tuple[str, ...]
    """Elements off their setpoint (full names)."""
    max_rel: float
    """Largest ``|S_delivered - S_set| / max(|S_set|, 1 VA)``."""
    worst: str
    """Element with the largest relative deviation."""

    def passes(self) -> bool:
        return not self.off


def _check_preconditions(ctx: DssContext, rel_tol: float) -> None:
    api = ctx.api
    with ctx.engine_call("Solution options"):
        mode = int(api.Solution.Mode())
        load_mult = float(api.Solution.LoadMult())
        gen_mult = float(api.Solution.GenMult())
        tolerance = float(api.Solution.Convergence())
    if mode != 0:
        raise ValueError(
            f"the setpoint check needs Snapshot mode (0), the solution is in mode {mode}"
        )
    if load_mult != 1.0 or gen_mult != 1.0:
        raise ValueError(
            f"the setpoint check needs LoadMult = GenMult = 1, got {load_mult}, {gen_mult}"
        )
    if tolerance * TOLERANCE_MARGIN > rel_tol:
        raise ValueError(
            f"solver Tolerance {tolerance:g} is too loose for a setpoint check at {rel_tol:g}: "
            f"use Tolerance <= {rel_tol / TOLERANCE_MARGIN:g}"
        )


def setpoint_report(ctx: DssContext, *, rel_tol: float = 1e-6) -> SetpointReport:
    """Check that every enabled load and generator of the solved circuit meets its setpoint."""
    _check_preconditions(ctx, rel_tol)
    api = ctx.api
    n_visited = 0
    names: list[str] = []
    s_set: list[complex] = []  # consumed VA
    powers: list[NDArray[np.float64]] = []  # interleaved P, Q (kW, kvar) per conductor
    with ctx.engine_call("Loads/Generators"):
        for cls, prefix, generator in (
            (api.Loads, "Load", False),
            (api.Generators, "Generator", True),
        ):
            i = cls.First()  # First and Next make the element the active circuit element
            while i > 0:
                n_visited += 1
                if api.CktElement.Enabled():
                    names.append(f"{prefix}.{cls.Name()}")
                    s = complex(cls.kW(), cls.kvar())
                    s_set.append((-s if generator else s) * 1e3)
                    powers.append(np.asarray(api.CktElement.Powers(), dtype=np.float64))
                i = cls.Next()
    if not names:
        return SetpointReport(n_checked=n_visited, off=(), max_rel=0.0, worst="")
    delivered = np.empty(len(powers), dtype=np.complex128)
    sizes = np.array([p.size for p in powers])
    for size in np.unique(sizes):  # rows of one length, each summed as numpy sums one element
        rows = np.flatnonzero(sizes == size)
        block = complex_from_interleaved(np.stack([powers[k] for k in rows]).ravel())
        delivered[rows] = block.reshape(rows.size, -1).sum(axis=1) * 1e3
    target = np.array(s_set)
    d = delivered - target  # hypot as Python's abs(complex); np.abs differs in the last bit
    rel = np.hypot(d.real, d.imag) / np.maximum(np.hypot(target.real, target.imag), 1.0)
    off = tuple(names[k] for k in np.flatnonzero(~(rel <= rel_tol)))  # NaN is off
    k = int(np.argmax(np.where(np.isnan(rel), np.inf, rel)))  # NaN is the worst
    return SetpointReport(n_checked=n_visited, off=off, max_rel=float(rel[k]), worst=names[k])
