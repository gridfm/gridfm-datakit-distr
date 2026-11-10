"""Topology helpers on terminal maps: galvanic zones and node reachability.

Pure functions on node bus names and elements, usable before the model is assembled.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from gridfm_datakit_distr.model.electrical import GROUND, Element


class UnionFind:
    """Disjoint sets over hashable string keys."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        """Sorted groups, sorted by their first member."""
        out: dict[str, list[str]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return sorted(sorted(group) for group in out.values())


def galvanic_zones(bus_of_node: Sequence[str], elements: Iterable[Element]) -> list[list[str]]:
    """Buses joined by enabled PD elements other than transformers (sorted groups of buses)."""
    zones = UnionFind(sorted(set(bus_of_node)))
    for element in elements:
        if not element.is_pd or not element.enabled or element.element_class == "transformer":
            continue
        nodes = [k for k in element.local_nodes() if k != GROUND]
        for k in nodes[1:]:
            zones.union(bus_of_node[nodes[0]], bus_of_node[k])
    return zones.groups()


def reachable_nodes(n_nodes: int, sources: Iterable[int], elements: Iterable[Element]) -> set[int]:
    """Nodes reachable from ``sources`` through enabled PD elements (transformers included)."""
    neighbours: list[set[int]] = [set() for _ in range(n_nodes)]
    for element in elements:
        if not element.is_pd or not element.enabled:
            continue
        nodes = [k for k in element.local_nodes() if k != GROUND]
        for k in nodes:
            neighbours[k].update(nodes)
    seen = set(sources)
    queue = deque(seen)
    while queue:
        for nxt in neighbours[queue.popleft()] - seen:
            seen.add(nxt)
            queue.append(nxt)
    return seen
