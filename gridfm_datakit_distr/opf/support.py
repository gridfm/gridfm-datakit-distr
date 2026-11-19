"""Which networks the OPF problem covers, and why the others are left out.

The problem of :mod:`gridfm_datakit_distr.opf.problem` covers networks whose power-delivery
elements (lines, switches, transformers, capacitors, reactors) enter through their admittances at
the network state, with one voltage source grounded at its second terminal, loads and PV units
(``Generator.pv_<load>``) connected wye (phases to a neutral node or to ground) or delta, on any
number of phases (:func:`~gridfm_datakit_distr.model.ground.unit_branches`). Other generators, PV
systems and storage are not covered yet. No CasADi here: the check runs without the ``opf`` extra.
"""

from __future__ import annotations

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.ground import floating_zones, is_conventional, unit_branches

PV_PREFIX = "generator.pv_"
"""PV units are the generators the variants place, ``Generator.pv_<load>``."""
PD_CLASSES = frozenset({"line", "switch", "transformer", "capacitor", "reactor"})
"""Power-delivery classes: they enter through their admittances at the network state."""
MAX_NAMES = 3


def unsupported_reason(model: ElectricalModel) -> str | None:
    """``None`` if the OPF problem covers ``model``, else why not (elements grouped by reason)."""
    reasons: dict[str, list[str]] = {}
    for element in model.elements:
        if element.enabled:
            why = _reason(model, element)
            if why is not None:
                reasons.setdefault(why, []).append(element.name)
    sources = sum(e.enabled and e.element_class == "vsource" for e in model.elements)
    if sources != 1:
        reasons.setdefault(f"{sources} voltage sources, the OPF needs exactly one", [])
    capacitive = sorted(
        {
            model.nodes[i].bus
            for z in floating_zones(model)
            if not is_conventional(model, z)
            for i in z
        }
    )
    if capacitive:  # zones under a numerical convention are covered on phase-to-phase voltages
        why = (
            "buses tied to ground only through line capacitance "
            "(OPF convergence there not established)"
        )
        reasons[why] = capacitive
    if not reasons:
        return None
    return "; ".join(_listed(why, names) for why, names in reasons.items())


def floating_buses(model: ElectricalModel) -> list[str]:
    """Buses of the galvanic zones that nothing ties to ground, sorted
    (:mod:`gridfm_datakit_distr.model.ground`)."""
    return sorted({model.nodes[i].bus for zone in floating_zones(model) for i in zone})


def _listed(why: str, names: list[str]) -> str:
    if not names:
        return why
    more = f" (+{len(names) - MAX_NAMES} more)" if len(names) > MAX_NAMES else ""
    return f"{why}: {', '.join(names[:MAX_NAMES])}{more}"


def _reason(model: ElectricalModel, element: Element) -> str | None:
    cls = element.element_class
    if cls in PD_CLASSES:
        return None
    if cls == "vsource":
        grounded = all(n == GROUND for terminal in element.terminals[1:] for n in terminal)
        return None if grounded else "voltage source not grounded at its second terminal"
    if cls == "generator" and not element.name.lower().startswith(PV_PREFIX):
        return "generator that is not a PV unit"
    if cls in ("load", "generator"):
        covered = unit_branches(model, element) is not None
        return None if covered else f"{cls} with a connection not covered by the OPF"
    return f"{cls} not covered by the OPF yet"
