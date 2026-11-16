"""Which networks the OPF problem covers, and why the others are left out.

The problem of :mod:`gridfm_datakit_distr.opf.problem` covers networks whose power-delivery
elements (lines, switches, transformers, capacitors, reactors) enter through their admittances at
the network state, with one voltage source grounded at its second terminal, loads and PV units
(``Generator.pv_<load>``) connected wye (phases to a neutral node or to ground) or delta, on any
number of phases (:func:`unit_branches`). Other generators, PV systems and storage are not covered
yet. No CasADi here: the check runs without the ``opf`` extra.
"""

from __future__ import annotations

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element

PV_PREFIX = "generator.pv_"
"""PV units are the generators the variants place, ``Generator.pv_<load>``."""
PD_CLASSES = frozenset({"line", "switch", "transformer", "capacitor", "reactor"})
"""Power-delivery classes: they enter through their admittances at the network state."""
MAX_NAMES = 3


def unit_branches(model: ElectricalModel, element: Element) -> list[tuple[int, int]] | None:
    """The branches (node pairs, :data:`GROUND` allowed in second place) a load or PV unit draws
    its power through, each an equal share of it; None for a connection not covered.

    One terminal of ``c`` conductors (OpenDSS ``NodeOrder``): wye when the last conductor is a
    neutral node or ground, one branch from each other conductor to it (P/n per phase, checked in
    spike s39); delta on three phases, ``(a, b), (b, c), (c, a)``; one branch between two phases.
    """
    if element.n_terminals != 1:
        return None
    nodes = element.local_nodes()

    def ref(node: int) -> bool:
        return node == GROUND or model.nodes[node].conductor == "n"

    if len(nodes) >= 2 and ref(nodes[-1]) and not any(ref(n) for n in nodes[:-1]):
        return [(n, nodes[-1]) for n in nodes[:-1]]
    if any(ref(n) for n in nodes):
        return None
    if len(nodes) == 3:
        return [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])]
    if len(nodes) == 2:
        return [(nodes[0], nodes[1])]
    return None


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
    floating = floating_buses(model)
    if floating:
        why = "buses without a ground reference (voltages to ground undetermined)"
        reasons[why] = floating
    if not reasons:
        return None
    return "; ".join(_listed(why, names) for why, names in reasons.items())


def floating_buses(model: ElectricalModel) -> list[str]:
    """Buses of the galvanic zones that nothing ties to ground, sorted.

    A zone gathers the nodes of every line, switch, reactor and capacitor, and of each terminal
    of a transformer (windings are coupled magnetically, not galvanically). It is grounded when it
    holds a source node, a grounded neutral, or an element position tied to node 0. In a floating
    zone (behind a delta-delta transformer, IEEE 37) only OpenDSS's anti-float admittances, about
    1e-6 of the series ones, set the common-mode voltage: a solve at ``tol = 1e-9`` cannot
    reproduce the voltages to ground within the verification's 1e-6 pu (T3.4).
    """
    parent = list(range(model.n_nodes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grounded = {n.idx for n in model.nodes if n.is_grounded or n.is_source}
    for e in model.elements:
        if not (e.enabled and e.is_pd):
            continue
        groups = e.terminals if e.element_class == "transformer" else [e.local_nodes()]
        for group in groups:
            nodes = [n for n in group if n != GROUND]
            if len(nodes) < len(group):
                grounded.update(nodes)
            for n in nodes[1:]:
                parent[find(n)] = find(nodes[0])
    roots = {find(i) for i in grounded}
    return sorted({n.bus for n in model.nodes if find(n.idx) not in roots})


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
