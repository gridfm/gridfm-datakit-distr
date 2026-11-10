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
    setpoints: list[tuple[str, complex]] = []  # (full name, consumed VA)
    with ctx.engine_call("Loads/Generators"):
        i = api.Loads.First()
        while i > 0:
            s = complex(api.Loads.kW(), api.Loads.kvar()) * 1e3
            setpoints.append((f"Load.{api.Loads.Name()}", s))
            i = api.Loads.Next()
        i = api.Generators.First()
        while i > 0:
            s = -complex(api.Generators.kW(), api.Generators.kvar()) * 1e3
            setpoints.append((f"Generator.{api.Generators.Name()}", s))
            i = api.Generators.Next()
    off: list[str] = []
    worst, max_rel = "", 0.0
    for name, s_set in setpoints:
        ctx.activate_element(name)
        with ctx.engine_call(f"CktElement.Powers {name}"):
            if not api.CktElement.Enabled():
                continue
            s = complex(complex_from_interleaved(api.CktElement.Powers()).sum()) * 1e3
        rel = abs(s - s_set) / max(abs(s_set), 1.0)
        if not rel <= rel_tol:  # NaN is off
            off.append(name)
        if rel > max_rel or not worst:
            worst, max_rel = name, rel
    return SetpointReport(n_checked=len(setpoints), off=tuple(off), max_rel=max_rel, worst=worst)
