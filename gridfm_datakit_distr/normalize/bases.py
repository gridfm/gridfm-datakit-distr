"""Voltage bases and the power base.

Every bus takes the line-to-neutral base of its galvanic zone (buses joined by PD elements other
than transformers). The nominal voltage of a zone comes from the ratings that feed it:

- the source zone: the Vsource ``basekV`` (line-to-line; line-to-neutral for a 1-phase source);
- any other zone: the transformer winding on it with the largest kVA (ties: definition order).
  ``kV`` is line-to-line for three-phase transformers; for single-phase ones it is the winding
  voltage: line-to-neutral for a wye winding (split-phase halves included), line-to-line for a
  delta winding.

``CalcVoltageBases`` rounds each bus to the nearest base of the ``VoltageBases`` list however far,
so the engine's ``kVBase`` is checked against the zone nominal:

- deviation <= ``CONSISTENT_REL``: consistent (nameplate rounding), kept;
- deviation > ``GROSS_REL``: the bus got another voltage level; :func:`fix_bases` re-bases it to
  the nominal with ``SetkVBase``;
- in between: ambiguous (a no-load secondary rating above the system nominal, or a close standard
  level such as 13.2/13.8 kV); reported, never decided automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel
from gridfm_datakit_distr.model.topology import galvanic_zones
from gridfm_datakit_distr.model.units import SQRT3

CONSISTENT_REL = 0.01
"""Nameplate roundings stay below 0.5 %: 2.4 kV vs 4.16/√3 kV is 0.07 %, 0.23 kV vs 0.4/√3 kV
is 0.4 %."""
GROSS_REL = 0.10
"""Nearest-base rounding to a wrong level: 0.48 kV for a 0.416 kV feeder is +15 %; a
0.24 kV list entry gives 0.24/√3 kV for a 0.12 kV split-phase half, +15.5 %. Legitimate no-load
secondary ratings stay below (433 V in a 400 V system is +8.3 %), but so do some wrong levels
(the default list puts 13.8 kV buses at 12.47 kV, -9.6 %): the band in between is ambiguous."""


class BaseError(RuntimeError):
    """A voltage base could not be set in the engine."""


@dataclass(frozen=True, slots=True)
class ZoneBase:
    """Nominal voltage of one galvanic zone."""

    buses: tuple[str, ...]
    nominal_kv_ln: float | None
    origin: str
    """The rating it comes from, e.g. ``Transformer.xfm1 winding 2``; empty without one."""


@dataclass(frozen=True, slots=True)
class BaseReport:
    """Engine voltage bases against the zone nominals."""

    zones: tuple[ZoneBase, ...]
    rebase: dict[str, float]
    """Bus -> nominal kV line-to-neutral, for buses off by more than ``GROSS_REL``."""
    ambiguous: tuple[str, ...]
    """Buses off by more than ``CONSISTENT_REL`` and at most ``GROSS_REL``."""
    conflicts: tuple[str, ...]
    """Zones with no rating, or with ratings that disagree by more than ``CONSISTENT_REL``."""
    max_deviation: float
    """Largest ``|kVBase / nominal - 1|`` over the buses of rated zones."""

    def passes(self) -> bool:
        return not self.rebase and not self.ambiguous and not self.conflicts

    def fix_commands(self) -> list[str]:
        """``SetkVBase`` for each bus to re-base (valid after loading, see :func:`fix_bases`)."""
        return [f"SetkVBase bus={bus} kVLN={kv!r}" for bus, kv in self.rebase.items()]


@dataclass(frozen=True, slots=True)
class _Rating:
    node: int
    kv_ln: float
    priority: float
    """``inf`` for a source, the transformer rating (VA) for a winding."""
    origin: str


def _ratings(ctx: DssContext, model: ElectricalModel) -> list[_Rating]:
    """The nominal voltage set by every source and transformer winding, in definition order."""
    api = ctx.api
    out: list[_Rating] = []
    for element in model.elements:
        if element.element_class == "vsource":
            ctx.activate_element(element.name)
            with ctx.engine_call(f"Vsources {element.name}"):
                api.Vsources.Name(element.name.split(".", 1)[1])
                kv, phases = float(api.Vsources.BasekV()), int(api.Vsources.Phases())
            node = next(k for k in element.terminals[0] if k != GROUND)
            out.append(_Rating(node, kv if phases == 1 else kv / SQRT3, math.inf, element.name))
        elif element.element_class == "transformer":
            ctx.activate_element(element.name)
            with ctx.engine_call(f"Transformers {element.name}"):
                phases = int(api.CktElement.NumPhases())
                api.Transformers.Name(element.name.split(".", 1)[1])
                for w, terminal in enumerate(element.terminals, start=1):
                    api.Transformers.Wdg(w)
                    kv, delta = float(api.Transformers.kV()), bool(api.Transformers.IsDelta())
                    nodes = [k for k in terminal if k != GROUND]
                    if not nodes:
                        continue
                    kv_ln = kv / SQRT3 if (phases > 1 or delta) else kv
                    priority = element.rating_va or 0.0
                    out.append(_Rating(nodes[0], kv_ln, priority, f"{element.name} winding {w}"))
    return out


def check_bases(ctx: DssContext, model: ElectricalModel) -> BaseReport:
    """Compare every bus's kVBase with the nominal voltage of its galvanic zone."""
    bus_of = [n.bus for n in model.nodes]
    groups = galvanic_zones(bus_of, model.elements)
    zone_of = {bus: i for i, group in enumerate(groups) for bus in group}
    feeds: dict[int, list[_Rating]] = {}
    for rating in _ratings(ctx, model):
        feeds.setdefault(zone_of[bus_of[rating.node]], []).append(rating)
    zones: list[ZoneBase] = []
    conflicts: list[str] = []
    nominal: dict[int, float] = {}
    for i, group in enumerate(groups):
        ratings = feeds.get(i, [])
        if not ratings:
            conflicts.append(f"zone {group[:3]}: no source or transformer rating")
            zones.append(ZoneBase(tuple(group), None, ""))
            continue
        chosen = max(ratings, key=lambda r: r.priority)  # first of equals: definition order
        for other in ratings:
            if abs(other.kv_ln / chosen.kv_ln - 1.0) > CONSISTENT_REL:
                conflicts.append(
                    f"zone {group[:3]}: {chosen.origin} {chosen.kv_ln:.4g} kV LN, "
                    f"{other.origin} {other.kv_ln:.4g} kV LN"
                )
        nominal[i] = chosen.kv_ln
        zones.append(ZoneBase(tuple(group), chosen.kv_ln, chosen.origin))

    rebase: dict[str, float] = {}
    ambiguous: list[str] = []
    worst = 0.0
    seen: set[str] = set()
    for n in model.nodes:
        zone = zone_of[n.bus]
        if n.bus in seen or zone not in nominal:
            continue
        seen.add(n.bus)
        signed = n.kv_base_ln / nominal[zone] - 1.0
        worst = max(worst, abs(signed))
        if abs(signed) > GROSS_REL:
            rebase[n.bus] = nominal[zone]
        elif abs(signed) > CONSISTENT_REL:
            ambiguous.append(
                f"bus {n.bus}: kVBase {n.kv_base_ln:.4g} kV LN, "
                f"nominal {nominal[zone]:.4g} ({signed:+.1%})"
            )
    return BaseReport(tuple(zones), rebase, tuple(ambiguous), tuple(conflicts), worst)


def fix_bases(ctx: DssContext, report: BaseReport) -> list[str]:
    """Re-base the buses of ``report.rebase`` in the engine; return the commands applied.

    Call it after :meth:`DssContext.load`: buses do not exist before the bus list is built, and
    ``CalcVoltageBases`` resets every base. ``SetkVBase`` on an unknown bus only
    prints a message, so every base is read back and a mismatch raises
    :class:`BaseError`. Bases do not enter the power-flow equations, only OpenDSS's stopping
    test, where a node's voltage change is measured in pu of its base.
    """
    api = ctx.api
    commands = report.fix_commands()
    for command, (bus, kv_ln) in zip(commands, report.rebase.items(), strict=True):
        ctx.command(command)
        with ctx.engine_call(f"Bus.kVBase {bus}"):
            found = int(api.Circuit.SetActiveBus(bus)) >= 0
            got = float(api.Bus.kVBase()) if found else math.nan
        if not math.isclose(got, kv_ln, rel_tol=1e-9):
            raise BaseError(f"{command}: bus {bus} has kVBase {got} kV after the command")
    return commands


def s_base_va(ctx: DssContext, model: ElectricalModel) -> tuple[float, str]:
    """The ``auto`` power base (VA) and where it comes from.

    The largest transformer with a winding in the source zone; without one (e.g. CSIRO feeders,
    whose source is at LV), the total nominal apparent power of the loads.
    """
    api = ctx.api
    bus_of = [n.bus for n in model.nodes]
    zone_of = {bus: i for i, g in enumerate(galvanic_zones(bus_of, model.elements)) for bus in g}
    source_zones = {zone_of[bus_of[k]] for k in model.source_nodes().tolist()}
    best: tuple[float, str] | None = None
    for element in model.elements:
        if element.element_class != "transformer":
            continue
        buses = {bus_of[k] for t in element.terminals for k in t if k != GROUND}
        rating = element.rating_va
        at_source = any(zone_of[b] in source_zones for b in buses)
        if at_source and rating is not None and (best is None or rating > best[0]):
            best = (rating, f"{element.name} rating")
    if best is not None:
        return best
    total = 0.0
    for element in model.elements:
        if element.element_class == "load":
            ctx.activate_element(element.name)
            with ctx.engine_call(f"Loads {element.name}"):
                api.Loads.Name(element.name.split(".", 1)[1])
                total += abs(complex(api.Loads.kW(), api.Loads.kvar())) * 1e3
    if total > 0:
        return total, "total nominal load"
    return 1e6, "default 1 MVA (no transformer at the source, no load)"
