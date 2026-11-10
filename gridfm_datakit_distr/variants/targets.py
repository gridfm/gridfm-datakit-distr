"""What a network offers to perturb, read once from its prepared base.

- Lines: every enabled line that is not a switch (switches keep their nominal impedance), with
  its length and unit, its series and shunt matrices per unit length (``Lines.RMatrix``,
  ``XMatrix``, ``CMatrix``: ohm and nF per the line's unit) and the positions of its neutral
  conductors.
- Grounding points (earth electrodes): wye windings grounded through ``Rneut >= 0`` and grounding
  reactors (neutral conductors to ground), except a reactor at a bus where a source has its star
  point on node 0 (a Vsource, a wye winding with its neutral on node 0): there the reactor bonds
  the neutral to the source and carries the whole return current (CSIRO's
  ``Reactor.grounding``), so its resistance is not an earthing parameter. Neutrals tied to node 0
  in a bus spec are solid by construction.
- Loads: enabled single-phase loads with one phase conductor (to neutral or ground) at a bus that
  has another phase; phase-to-phase loads are not moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.extract.extract_semantic import is_grounding_reactor
from gridfm_datakit_distr.model.electrical import GROUND, ComplexMatrix
from gridfm_datakit_distr.networks.prepare import PreparedNetwork

_PHASES = ("a", "b", "c")


@dataclass(frozen=True, slots=True, eq=False)
class LineTarget:
    element: str
    """Full name, ``Line.<name>``."""
    length: float
    """In the line's unit."""
    units: str
    """OpenDSS text of the line's length unit (``m``, ``km``, ``none``, ...)."""
    neutral: tuple[int, ...]
    """Conductor positions of the neutral conductors."""
    r: NDArray[np.float64]
    """Series resistance per unit length (ohm), conductor order."""
    x: NDArray[np.float64]
    """Series reactance per unit length (ohm)."""
    c: NDArray[np.float64]
    """Shunt capacitance per unit length (nF)."""
    yprim: ComplexMatrix
    """Yprim of the unperturbed line (S)."""

    @property
    def name(self) -> str:
        return self.element.split(".", 1)[1]


@dataclass(frozen=True, slots=True)
class GroundingTarget:
    key: str
    """``Reactor.<name>`` or ``Transformer.<name>/w<winding>``."""
    command: str
    """Edit command up to the new resistance, e.g. ``Edit Transformer.t wdg=2 Rneut=``."""
    r_ohm: float
    x_ohm: float


@dataclass(frozen=True, slots=True)
class LoadTarget:
    element: str
    """Full name, ``Load.<name>``."""
    bus: str
    phase: str
    """OpenDSS node of its phase conductor."""
    rest: tuple[str, ...]
    """Its other conductors (the neutral); empty when it returns through ground implicitly."""
    candidates: tuple[str, ...]
    """The other phase nodes of its bus."""

    @property
    def name(self) -> str:
        return self.element.split(".", 1)[1]

    def bus1(self, phase: str) -> str:
        return ".".join([self.bus, phase, *self.rest])


@dataclass(frozen=True, slots=True)
class CustomerTarget:
    """An enabled load and its connection, where a PV unit can be placed."""

    element: str
    bus1: str
    """As the engine reports it (``BusNames``), nodes included when given."""
    phases: int
    kv: float
    conn: str
    """``wye`` or ``delta``."""

    @property
    def name(self) -> str:
        return self.element.split(".", 1)[1]


@dataclass(frozen=True, slots=True)
class VariantTargets:
    lines: tuple[LineTarget, ...]
    groundings: tuple[GroundingTarget, ...]
    loads: tuple[LoadTarget, ...]
    """Movable single-phase loads (phase reallocation)."""
    customers: tuple[CustomerTarget, ...] = ()
    """Every enabled load (PV placement)."""


def collect_targets(prepared: PreparedNetwork) -> VariantTargets:
    """The perturbation targets of a prepared (unperturbed) network, each kind sorted by name."""
    return VariantTargets(
        _lines(prepared), _groundings(prepared), _loads(prepared), _customers(prepared)
    )


def _customers(prepared: PreparedNetwork) -> tuple[CustomerTarget, ...]:
    api = prepared.ctx.api
    out = []
    i = api.Loads.First()
    while i:
        if api.CktElement.Enabled():
            out.append(
                CustomerTarget(
                    element=f"Load.{api.Loads.Name()}",
                    bus1=str(api.CktElement.BusNames()[0]),
                    phases=int(api.CktElement.NumPhases()),
                    kv=float(api.Loads.kV()),
                    conn="delta" if api.Loads.IsDelta() else "wye",
                )
            )
        i = api.Loads.Next()
    return tuple(sorted(out, key=lambda c: c.element))


def _lines(prepared: PreparedNetwork) -> tuple[LineTarget, ...]:
    ctx, model = prepared.ctx, prepared.model
    api = ctx.api
    out = []
    for e in sorted(model.elements, key=lambda e: e.name):
        if e.element_class != "line" or not e.enabled or e.yprim is None:
            continue
        name = e.name.split(".", 1)[1]
        api.Lines.Name(name)
        n = len(e.terminals[0])
        neutral = tuple(
            p
            for p, k in enumerate(e.terminals[0])
            if k != GROUND and model.nodes[k].conductor == "n"
        )
        out.append(
            LineTarget(
                element=e.name,
                length=float(api.Lines.Length()),
                units=ctx.command(f"? Line.{name}.units").strip(),
                neutral=neutral,
                r=np.asarray(api.Lines.RMatrix(), dtype=np.float64).reshape(n, n),
                x=np.asarray(api.Lines.XMatrix(), dtype=np.float64).reshape(n, n),
                c=np.asarray(api.Lines.CMatrix(), dtype=np.float64).reshape(n, n),
                yprim=e.yprim,
            )
        )
    return tuple(out)


def _star_ground_buses(prepared: PreparedNetwork) -> set[str]:
    """Buses where a source has its star point on node 0: a Vsource whose second terminal is
    ground, a wye winding (``Rneut < 0``) whose neutral conductor is node 0."""
    model, api = prepared.model, prepared.ctx.api
    elements = {e.name: e for e in model.elements if e.enabled}
    buses: set[str] = set()
    for e in elements.values():
        if e.element_class == "vsource" and all(k == GROUND for t in e.terminals[1:] for k in t):
            buses |= {model.nodes[k].bus for k in e.terminals[0] if k != GROUND}
    i = api.Transformers.First()
    while i:
        element = elements.get(f"Transformer.{api.Transformers.Name()}")
        if element is not None:
            for w in range(1, api.Transformers.NumWindings() + 1):
                api.Transformers.Wdg(w)
                terminal = element.terminals[w - 1]
                if (
                    not api.Transformers.IsDelta()
                    and api.Transformers.Rneut() < 0
                    and terminal[-1] == GROUND
                ):
                    buses |= {model.nodes[k].bus for k in terminal if k != GROUND}
        i = api.Transformers.Next()
    return buses


def _groundings(prepared: PreparedNetwork) -> tuple[GroundingTarget, ...]:
    model, api = prepared.model, prepared.ctx.api
    names = {e.name for e in model.elements if e.enabled}
    bonds = _star_ground_buses(prepared)
    out = []
    for e in model.elements:
        if e.element_class == "reactor" and e.enabled and is_grounding_reactor(e, model):
            if {model.nodes[k].bus for k in e.terminals[0] if k != GROUND} & bonds:
                continue  # the neutral's bond to a source's star point, not an electrode
            name = e.name.split(".", 1)[1]
            api.Reactors.Name(name)
            r, x = float(api.Reactors.R()), float(api.Reactors.X())
            out.append(GroundingTarget(e.name, f"Edit {e.name} R=", r, x))
    i = api.Transformers.First()
    while i:
        name = api.Transformers.Name()
        if f"Transformer.{name}" in names:
            for w in range(1, api.Transformers.NumWindings() + 1):
                api.Transformers.Wdg(w)
                if not api.Transformers.IsDelta() and api.Transformers.Rneut() >= 0:
                    out.append(
                        GroundingTarget(
                            f"Transformer.{name}/w{w}",
                            f"Edit Transformer.{name} wdg={w} Rneut=",
                            float(api.Transformers.Rneut()),
                            float(api.Transformers.Xneut()),
                        )
                    )
        i = api.Transformers.Next()
    return tuple(sorted(out, key=lambda g: g.key))


def _loads(prepared: PreparedNetwork) -> tuple[LoadTarget, ...]:
    model, api = prepared.model, prepared.ctx.api
    phases_at: dict[str, set[str]] = {}
    for node in model.nodes:
        if node.conductor in _PHASES:
            phases_at.setdefault(node.bus, set()).add(node.opendss_node.rsplit(".", 1)[1])
    out = []
    i = api.Loads.First()
    while i:
        if api.CktElement.Enabled() and api.CktElement.NumPhases() == 1:
            bus = api.CktElement.BusNames()[0].split(".")[0].lower()
            nodes = [str(int(n)) for n in api.CktElement.NodeOrder()]
            available = phases_at.get(bus, set())
            phase_nodes = [n for n in nodes if n in available]
            if len(phase_nodes) == 1 and len(available) > 1:
                (phase,) = phase_nodes
                out.append(
                    LoadTarget(
                        element=f"Load.{api.Loads.Name()}",
                        bus=bus,
                        phase=phase,
                        rest=tuple(n for n in nodes if n not in (phase, "0")),
                        candidates=tuple(sorted(available - {phase}, key=int)),
                    )
                )
        i = api.Loads.Next()
    return tuple(sorted(out, key=lambda t: t.element))
