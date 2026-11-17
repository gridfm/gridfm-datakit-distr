"""Galvanic zones without a ground reference, and what sets their voltage to ground.

A zone gathers the nodes of every line, switch, reactor and capacitor, and of each terminal of a
transformer (windings are coupled magnetically, not galvanically). It is grounded when it holds a
source node, a grounded neutral, or an element position tied to node 0. In a floating zone the
common-mode voltage hangs on shunt admittances alone, weak against the series ones. Two kinds are
told apart by where that shunt comes from (spike s43):

- *conventional*: no modelled element but the transformers has a shunt to ground in the zone; what
  remains is OpenDSS's anti-float admittance of the windings, a numerical convention (IEEE 123's
  bus 610: 6.5e-7 S). The voltage to ground there is not a label; voltage differences are.
- *capacitive*: lines or other elements carry a shunt to ground (cable capacitance; behind IEEE
  37's delta-delta substation 8.7e-4 S, against 2.6e-6 S of anti-float terms): physical, weakly
  coupled.
"""

from __future__ import annotations

import numpy as np

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element

SHUNT_REL = 1e-12
"""A zone's shunt to ground from elements other than transformers counts when ``|1ᵀ Y_ZZ 1|`` of
those elements exceeds this share of the zone's largest Yprim entry (series-only elements sum to
round-off; IEEE 37's cables give 8.7e-7)."""


def floating_zones(model: ElectricalModel) -> list[list[int]]:
    """Node indices of each galvanic zone that nothing ties to ground (sorted, by first node)."""
    parent = list(range(model.n_nodes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grounded = {node.idx for node in model.nodes if node.is_grounded or node.is_source}
    for e in model.elements:
        if not (e.enabled and e.is_pd):
            continue
        local = e.local_nodes()
        tied = tied_to_ground(e)
        groups = (
            [
                list(range(k * e.n_conductors, (k + 1) * e.n_conductors))
                for k in range(e.n_terminals)
            ]
            if e.element_class == "transformer"
            else [list(range(len(local)))]
        )
        for positions in groups:
            nodes = [local[k] for k in positions if local[k] != GROUND]
            if any(local[k] == GROUND and tied[k] for k in positions):
                grounded.update(nodes)
            for n in nodes[1:]:
                parent[find(n)] = find(nodes[0])
    roots = {find(i) for i in grounded}
    zones: dict[int, list[int]] = {}
    for node in model.nodes:
        if find(node.idx) not in roots:
            zones.setdefault(find(node.idx), []).append(node.idx)
    return sorted(zones.values())


def is_conventional(model: ElectricalModel, zone: list[int]) -> bool:
    """Whether only transformers' anti-float terms tie ``zone`` to ground (see module doc)."""
    inside = set(zone)
    shunt, scale = 0.0 + 0.0j, 0.0
    for e in model.elements:
        if not (e.enabled and e.is_pd) or e.yprim is None:
            continue
        k = [i for i, n in enumerate(e.local_nodes()) if n != GROUND and n in inside]
        if not k:
            continue
        block = e.yprim[np.ix_(k, k)]
        scale = max(scale, float(np.abs(block).max()))
        if e.element_class != "transformer":
            shunt += complex(block.sum())
    return abs(shunt) <= SHUNT_REL * scale


def conventional_zones(model: ElectricalModel) -> list[list[int]]:
    """The floating zones whose voltage to ground only a numerical convention sets."""
    return [z for z in floating_zones(model) if is_conventional(model, z)]


def tied_to_ground(element: Element) -> list[bool]:
    """Per local position: whether the Yprim couples it (a delta winding lists an unused fourth
    position at node 0, with an empty row)."""
    if element.yprim is None:
        return [False] * element.n_positions
    rows = np.abs(element.yprim).max(axis=1)
    return [bool(r >= 1e-3 * rows.max()) for r in rows]
