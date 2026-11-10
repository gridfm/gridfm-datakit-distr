"""Neutral representation and grounding of a network.

A *zone* is a set of buses joined galvanically, i.e. by any PD element except transformers.
A zone is

- ``explicit`` if it has neutral nodes (conductor ``n``): the neutral is a modelled conductor;
- ``kron`` if it has no neutral node but some load, generator, PV or storage ties a conductor
  to ground (node 0): the neutral is assumed at ground potential everywhere (Kron-reduced);
- ``three_wire`` otherwise (delta-connected equipment only): compatible with both.

The network is ``explicit`` if it has explicit zones and no kron zone, ``mixed`` if it has
both, and ``kron`` otherwise (no neutral nodes). Grounded neutral nodes come from the
extractor (admittance to ground ≥ ``GROUNDING_MIN_S``); a set of neutral nodes joined by lines
or switches with no grounded node is a *floating neutral island*.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel
from gridfm_datakit_distr.model.topology import UnionFind, galvanic_zones

NeutralRepresentation = Literal["explicit", "kron", "mixed"]
ZoneKind = Literal["explicit", "kron", "three_wire"]


class NeutralError(ValueError):
    """The neutral representation is not acceptable (mixed, floating, or not as expected)."""


@dataclass(frozen=True, slots=True)
class Zone:
    """Buses joined galvanically, and how their neutral is represented."""

    buses: tuple[str, ...]
    kind: ZoneKind


@dataclass(frozen=True, slots=True)
class NeutralReport:
    representation: NeutralRepresentation
    zones: tuple[Zone, ...]
    grounded_neutrals: dict[str, complex]
    """Grounded neutral node (OpenDSS name) -> grounding impedance (ohm)."""
    floating_islands: tuple[tuple[str, ...], ...]
    """Neutral nodes joined by lines/switches with no grounded node among them."""
    grounded_pc_in_explicit: tuple[str, ...]
    """PC elements tying a conductor to ground inside an explicit zone (local earthing)."""


def classify_neutral(model: ElectricalModel) -> NeutralReport:
    """Classify the neutral representation of ``model`` and find its groundings."""
    bus_of = [node.bus for node in model.nodes]
    islands = UnionFind([n.opendss_node for n in model.nodes if n.conductor == "n"])
    for element in model.elements:
        if not element.is_pd or element.element_class == "transformer":
            continue
        nodes = [k for k in element.local_nodes() if k != GROUND]
        neutrals = [model.nodes[k].opendss_node for k in nodes if model.nodes[k].conductor == "n"]
        for left, right in itertools.pairwise(neutrals):
            islands.union(left, right)
    groups = galvanic_zones(bus_of, model.elements)
    zone_of = {bus: i for i, group in enumerate(groups) for bus in group}
    has_neutral = {zone_of[n.bus] for n in model.nodes if n.conductor == "n"}
    grounded_pc: dict[int, list[str]] = {}
    for element in model.elements:
        if element.is_pd or element.element_class == "vsource":
            continue
        local = element.local_nodes()
        if GROUND in local:
            zone = zone_of[bus_of[next(k for k in local if k != GROUND)]]
            grounded_pc.setdefault(zone, []).append(element.name)

    kinds: list[Zone] = []
    for i, group in enumerate(groups):
        kind: ZoneKind = (
            "explicit" if i in has_neutral else ("kron" if i in grounded_pc else "three_wire")
        )
        kinds.append(Zone(tuple(group), kind))
    kind_set = {z.kind for z in kinds}
    representation: NeutralRepresentation = (
        "mixed"
        if {"explicit", "kron"} <= kind_set
        else "explicit"
        if "explicit" in kind_set
        else "kron"
    )

    grounded = {
        n.opendss_node: complex(n.r_ground_ohm or 0.0, n.x_ground_ohm or 0.0)
        for n in model.nodes
        if n.is_grounded
    }
    floating = tuple(
        tuple(island) for island in islands.groups() if not any(name in grounded for name in island)
    )
    in_explicit = tuple(
        sorted(name for zone, names in grounded_pc.items() if zone in has_neutral for name in names)
    )
    return NeutralReport(representation, tuple(kinds), grounded, floating, in_explicit)


def check_neutral(
    report: NeutralReport,
    expected: NeutralRepresentation | None = None,
    *,
    allow_mixed: bool = False,
    allow_floating: bool = False,
) -> None:
    """Raise :class:`NeutralError` for a mixed representation, floating neutral islands, or a
    representation other than ``expected`` (check C6: the catalog expectation). A catalog that
    expects ``mixed`` accepts it."""
    allow_mixed = allow_mixed or expected == "mixed"
    if expected is not None and report.representation != expected:
        raise NeutralError(
            f"neutral representation is {report.representation}, expected {expected}"
        )
    if report.representation == "mixed" and not allow_mixed:
        zones = [z.buses[:3] for z in report.zones if z.kind != "three_wire"]
        raise NeutralError(f"mixed neutral representation (explicit and Kron zones): {zones[:6]}")
    if report.floating_islands and not allow_floating:
        sizes = [len(island) for island in report.floating_islands]
        examples = [island[:3] for island in report.floating_islands[:3]]
        raise NeutralError(
            f"{len(sizes)} floating neutral islands ({sum(sizes)} nodes), e.g. {examples}"
        )
