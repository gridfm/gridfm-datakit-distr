"""Conductor labels and phase consistency.

Labels follow the OpenDSS node numbers (1, 2, 3 -> a, b, c; 4 -> n), except in split-phase
zones: the galvanic zone of the secondary of a centre-tap transformer, where the two outer
nodes are ``s1`` and ``s2`` and the centre (if it is a node, not ground) is ``n``. A centre-tap
transformer is a single-phase transformer with three windings whose second and third windings
share a node: ``buses=[p.1 s.1.c s.c.2]`` (``c`` = 0 for a grounded centre).
"""

from __future__ import annotations

from dataclasses import dataclass

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.topology import galvanic_zones, reachable_nodes

DEFAULT_LABELS = {1: "a", 2: "b", 3: "c", 4: "n"}


class PhaseError(ValueError):
    """Conductor labels cannot be assigned, or the phases are inconsistent (strict mode)."""


@dataclass(frozen=True, slots=True)
class CentreTap:
    """The outer nodes and the centre of a split-phase secondary."""

    transformer: str
    s1: int
    s2: int
    centre: int
    """Node index of the centre, or ``GROUND``."""


def centre_taps(elements: list[Element]) -> list[CentreTap]:
    """Centre-tap transformers, recognised from their terminal maps."""
    taps = []
    for e in elements:
        if e.element_class != "transformer" or e.n_terminals != 3 or e.n_conductors != 2:
            continue
        (p, c2), (c3, q) = e.terminals[1], e.terminals[2]
        if c2 == c3 and GROUND not in (p, q) and p != q:
            taps.append(CentreTap(e.name, s1=p, s2=q, centre=c2))
    return taps


def conductor_labels(node_names: list[str], elements: list[Element]) -> list[str]:
    """Label every node (``bus.k`` names in model order); raise on unknown patterns."""
    bus_of = [name.rpartition(".")[0] for name in node_names]
    number = [int(name.rpartition(".")[2]) for name in node_names]
    zone_of = {bus: i for i, group in enumerate(galvanic_zones(bus_of, elements)) for bus in group}
    split: dict[int, dict[int, str]] = {}
    for tap in centre_taps(elements):
        mapping = {number[tap.s1]: "s1", number[tap.s2]: "s2"}
        if tap.centre != GROUND:
            mapping[number[tap.centre]] = "n"
        zone = zone_of[bus_of[tap.s1]]
        if split.setdefault(zone, mapping) != mapping:
            raise PhaseError(f"{tap.transformer}: split-phase zone fed with different node numbers")
    labels = []
    for name, bus, k in zip(node_names, bus_of, number, strict=True):
        table = split.get(zone_of[bus], DEFAULT_LABELS)
        if k not in table:
            raise PhaseError(f"node {name}: conductor {k} has no label in its zone")
        labels.append(table[k])
    return labels


@dataclass(frozen=True, slots=True)
class PhaseReport:
    mismatches: tuple[str, ...]
    """Series elements joining conductors with different labels (e.g. a lateral from b to a)."""
    islanded: tuple[str, ...]
    """Nodes not reachable from the source through PD elements."""

    def passes(self) -> bool:
        return not self.mismatches and not self.islanded


def phase_consistency(model: ElectricalModel) -> PhaseReport:
    """Check that series elements keep phases and that every node is fed from the source."""
    labels = [n.conductor for n in model.nodes]
    names = [n.opendss_node for n in model.nodes]
    mismatches = []
    for e in model.elements:
        if not e.is_pd or e.element_class == "transformer" or e.n_terminals != 2:
            continue
        for a, b in zip(e.terminals[0], e.terminals[1], strict=True):
            if GROUND not in (a, b) and labels[a] != labels[b]:
                mismatches.append(f"{e.name}: {names[a]} ({labels[a]}) -> {names[b]} ({labels[b]})")
    reached = reachable_nodes(model.n_nodes, model.source_nodes().tolist(), model.elements)
    islanded = tuple(names[k] for k in range(model.n_nodes) if k not in reached)
    return PhaseReport(tuple(mismatches), islanded)


def check_phases(report: PhaseReport, *, strict: bool = False) -> list[str]:
    """Return the problems as messages; raise :class:`PhaseError` if ``strict`` and any."""
    problems = [*report.mismatches, *(f"islanded node {n}" for n in report.islanded)]
    if strict and problems:
        raise PhaseError(f"{len(problems)} phase problems, e.g. {problems[:5]}")
    return problems
