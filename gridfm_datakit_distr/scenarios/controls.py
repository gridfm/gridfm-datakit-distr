"""Controls per scenario: settle and freeze the network state.

``static_settle``: with the scenario's setpoints applied, ``ControlMode=STATIC`` lets the
regulators and capacitor controls settle within one solve. With a tap jitter ``J``, the taps of the
regulated windings then move by a uniform integer number of steps in ``[-J, J]`` (``TAP_JITTER``
stream), clipped to the tap limits, and the controls settle again from there: a regulator does not
move while its voltage is inside its band, so where it stops depends on where it starts, and every
state is one the controls hold. The controls are frozen (``ControlMode=OFF``) at that state, which
the label solve uses (labels always use ``OFF``). ``as_is``: ``OFF`` with the taps and capacitor
states as compiled. A frozen state reproduces the settled solution (C3).

The state covers the transformers of the RegControls (all their windings) and the capacitors of the
CapControls: the tap-dependent elements of the model (``Element.is_tap_dependent``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pyarrow as pa

from gridfm_datakit_distr.engine.context import DssContext, SolveStatus
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.state import NetworkState, Origin

ControlsMode = Literal["static_settle", "as_is"]
MAX_CONTROL_ITER = 100


def regulated_windings(ctx: DssContext) -> list[tuple[str, int, float, float, float]]:
    """(transformer, winding, min tap, max tap, step) of every RegControl, ordered."""
    api = ctx.api
    pairs = set()
    i = api.RegControls.First()
    while i:
        pairs.add((str(api.RegControls.Transformer()).lower(), int(api.RegControls.Winding())))
        i = api.RegControls.Next()
    out = []
    for t, w in sorted(pairs):
        api.Transformers.Name(t)
        api.Transformers.Wdg(w)
        lo, hi = float(api.Transformers.MinTap()), float(api.Transformers.MaxTap())
        out.append((t, w, lo, hi, (hi - lo) / int(api.Transformers.NumTaps())))
    return out


def _controlled_capacitors(ctx: DssContext) -> list[str]:
    api, names = ctx.api, set()
    i = api.CapControls.First()
    while i:
        names.add(str(api.CapControls.Capacitor()).lower())
        i = api.CapControls.Next()
    return sorted(names)


def read_state(ctx: DssContext, variant_id: str, origin: Origin) -> NetworkState:
    """The taps of the regulated transformers and the steps of the controlled capacitors."""
    api = ctx.api
    taps = []
    for t in sorted({t for t, *_ in regulated_windings(ctx)}):
        api.Transformers.Name(t)
        windings = []
        for w in range(1, int(api.Transformers.NumWindings()) + 1):
            api.Transformers.Wdg(w)
            windings.append(float(api.Transformers.Tap()))
        taps.append((t, tuple(windings)))
    capacitors = []
    for c in _controlled_capacitors(ctx):
        api.Capacitors.Name(c)
        capacitors.append((c, tuple(int(s) for s in api.Capacitors.States())))
    return NetworkState(variant_id, tuple(taps), tuple(capacitors), origin)


def apply_state(ctx: DssContext, state: NetworkState) -> None:
    """Set the taps and capacitor steps of ``state`` (taken at the next solve)."""
    api = ctx.api
    for t, taps in state.taps:
        api.Transformers.Name(t)
        for w, tap in enumerate(taps, start=1):
            api.Transformers.Wdg(w)
            api.Transformers.Tap(float(tap))
    for c, states in state.capacitors:
        api.Capacitors.Name(c)
        api.Capacitors.States(list(states))


def settle(
    ctx: DssContext,
    mode: ControlsMode,
    variant_id: str,
    *,
    jitter_steps: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[NetworkState, SolveStatus | None]:
    """Settle (or keep) the controls of the setpoints in ``ctx`` and freeze them: the returned
    state is applied and ``ControlMode`` is ``OFF``; the status is the last settling solve's. The
    origin is ``static_settle+jitter`` when settling again from the jittered taps ends at another
    state than the first settling."""
    if mode == "as_is":
        ctx.command("Set ControlMode=OFF")
        return read_state(ctx, variant_id, "as_is"), None
    ctx.command(f"Set ControlMode=STATIC MaxControlIter={MAX_CONTROL_ITER}")
    status = ctx.solve()
    state = read_state(ctx, variant_id, "static_settle")
    if jitter_steps and status.converged:
        if rng is None:
            raise ValueError("tap jitter needs the scenario's TAP_JITTER generator")
        apply_state(ctx, _jitter(ctx, state, jitter_steps, rng))
        status = ctx.solve()  # the controls settle again from the jittered taps
        again = read_state(ctx, variant_id, "static_settle+jitter")
        if again.state_id != state.state_id:
            state = again
    ctx.command("Set ControlMode=OFF")
    apply_state(ctx, state)
    return state, status


def _jitter(
    ctx: DssContext, state: NetworkState, steps: int, rng: np.random.Generator
) -> NetworkState:
    """``state`` with each regulated tap moved by a uniform integer in ``[-steps, steps]`` tap
    steps, clipped to the tap limits."""
    taps = {t: list(values) for t, values in state.taps}
    for t, w, lo, _hi, step in regulated_windings(ctx):
        n = round((_hi - lo) / step)
        position = round((taps[t][w - 1] - lo) / step)
        new = min(max(position + int(rng.integers(-steps, steps + 1)), 0), n)
        if new != position:
            taps[t][w - 1] = lo + new * step
    return replace(state, taps=tuple((t, tuple(v)) for t, v in sorted(taps.items())))


def model_at_state(model: ElectricalModel, ctx: DssContext) -> ElectricalModel:
    """``model`` with the Yprims of its tap-dependent elements read from ``ctx`` (after a
    solve)."""
    elements = [
        replace(e, yprim=ctx.element_yprim(e.name))
        if e.is_tap_dependent and e.yprim is not None
        else e
        for e in model.elements
    ]
    return ElectricalModel(model.network_id, model.variant_id, list(model.nodes), elements)


@dataclass(slots=True)
class StateRegistry:
    """The distinct states of a variant's scenarios, with how many scenarios use each."""

    states: dict[str, NetworkState] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, state: NetworkState) -> str:
        sid = state.state_id
        self.states.setdefault(sid, state)  # every route to a state gives the same content
        self.counts[sid] = self.counts.get(sid, 0) + 1
        return sid

    def table(self) -> pa.Table:
        """Rows of ``states/states.parquet`` (outputs.md §5)."""
        rows = [
            {
                "state_id": sid,
                "variant_id": s.variant_id,
                "taps": s.taps_json(),
                "capacitor_states": s.capacitors_json(),
                "n_scenarios": self.counts[sid],
            }
            for sid, s in self.states.items()
        ]
        return pa.Table.from_pylist(rows, schema=schema.STATES)
