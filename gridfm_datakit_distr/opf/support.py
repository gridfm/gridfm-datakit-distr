"""Which networks the OPF problem covers, and why the others are left out.

The problem of :mod:`gridfm_datakit_distr.opf.problem` covers networks made of lines, a
neutral-grounding reactor and one voltage source grounded at its second terminal, with single-phase
phase-to-neutral loads and PV units (``Generator.pv_<load>``) only. Transformers, capacitors,
switches, other reactors, delta or multi-phase devices, other generators, PV systems and storage
are not covered yet. No CasADi here: the check runs without the ``opf`` extra.
"""

from __future__ import annotations

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element

PHASES = frozenset({"a", "b", "c"})
PV_PREFIX = "generator.pv_"
"""PV units are the generators the variants place, ``Generator.pv_<load>``."""
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
    if not reasons:
        return None
    return "; ".join(_listed(why, names) for why, names in reasons.items())


def _listed(why: str, names: list[str]) -> str:
    if not names:
        return why
    more = f" (+{len(names) - MAX_NAMES} more)" if len(names) > MAX_NAMES else ""
    return f"{why}: {', '.join(names[:MAX_NAMES])}{more}"


def _reason(model: ElectricalModel, element: Element) -> str | None:
    cls = element.element_class
    if cls == "line":
        return None
    if cls == "reactor":
        nodes = element.local_nodes()
        to_neutral = all(n == GROUND or model.nodes[n].conductor == "n" for n in nodes)
        return None if to_neutral and GROUND in nodes else "reactor other than a neutral grounding"
    if cls == "vsource":
        grounded = all(n == GROUND for terminal in element.terminals[1:] for n in terminal)
        return None if grounded else "voltage source not grounded at its second terminal"
    if cls == "generator" and not element.name.lower().startswith(PV_PREFIX):
        return "generator that is not a PV unit"
    if cls in ("load", "generator"):
        return (
            None
            if _phase_to_neutral(model, element)
            else f"{cls} not single-phase phase-to-neutral"
        )
    return f"{cls} not covered by the OPF yet"


def _phase_to_neutral(model: ElectricalModel, element: Element) -> bool:
    """One terminal joining one phase node and the neutral node of the same bus."""
    nodes = element.local_nodes()
    if element.n_terminals != 1 or len(nodes) != 2 or GROUND in nodes:
        return False
    a, b = (model.nodes[n] for n in nodes)
    pair = {a.conductor, b.conductor}
    return a.bus == b.bus and len(pair) == 2 and "n" in pair and pair - {"n"} <= PHASES
