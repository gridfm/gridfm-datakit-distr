"""Electrical-layer extractor: compiled OpenDSS circuit → :class:`ElectricalModel`.

Everything is read back from the engine API after compilation; the API facts relied on were
confirmed by experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.classes import (
    CONTROL_CLASSES,
    PC_CLASSES,
    PD_CLASSES,
    SOURCE_CLASSES,
)
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element, Node
from gridfm_datakit_distr.normalize.phases import PhaseError, conductor_labels

log = logging.getLogger(__name__)

GROUNDING_MIN_S = 1e-3
"""A neutral node is grounded if its admittance to ground is at least this (S): 1 kOhm or less.
Anti-float shunts (~1e-6 S) and cable charging of LV feeders stay well below."""


class ExtractionError(RuntimeError):
    """The compiled circuit cannot be represented by the electrical model."""


class UnsupportedElementError(ExtractionError):
    """Enabled elements of classes the model does not support."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(f"{len(names)} enabled elements of unsupported classes: {names[:10]}")


@dataclass(frozen=True, slots=True)
class _Raw:
    """What the engine says about one element."""

    name: str
    element_class: str
    buses: list[str]
    node_order: list[int]
    n_terminals: int
    n_conductors: int


def _enabled_pd_names(ctx: DssContext) -> set[str]:
    """Lower-case names from ``First/NextPDElement``, which skips disabled elements."""
    api = ctx.api
    names = set()
    i = api.Circuit.FirstPDElement()
    while i > 0:
        names.add(str(api.CktElement.Name()).lower())
        i = api.Circuit.NextPDElement()
    return names


def _raw_elements(ctx: DssContext) -> tuple[list[_Raw], list[tuple[str, str]]]:
    """Enabled elements as the engine reports them, and (name, class) of the disabled ones.

    ``NodeOrder()`` of a disabled element raises (#15013, nodes not initialised), so disabled
    elements have no reliable terminal map and are left out of the electrical model.
    """
    api = ctx.api
    enabled: list[_Raw] = []
    disabled: list[tuple[str, str]] = []
    for name in api.Circuit.AllElementNames():
        ctx.activate_element(name)
        ce = api.CktElement
        element_class = name.split(".", 1)[0].lower()
        with ctx.engine_call(f"read {name}"):
            if not ce.Enabled():
                disabled.append((str(ce.Name()), element_class))
                continue
            enabled.append(
                _Raw(
                    name=str(ce.Name()),
                    element_class=element_class,
                    buses=[b.split(".")[0].lower() for b in ce.BusNames()],
                    node_order=[int(k) for k in ce.NodeOrder()],
                    n_terminals=int(ce.NumTerminals()),
                    n_conductors=int(ce.NumConductors()),
                )
            )
    return enabled, disabled


def disabled_elements(ctx: DssContext) -> list[str]:
    """Names of the disabled elements of the compiled circuit (left out of the model)."""
    return [name for name, _ in _raw_elements(ctx)[1]]


def _nodes(ctx: DssContext) -> tuple[list[Node], dict[str, int]]:
    api = ctx.api
    names = ctx.ynode_order()
    kv_base: dict[str, float] = {}
    nodes = []
    for idx, name in enumerate(names):
        bus = name.rpartition(".")[0]
        if bus not in kv_base:
            api.Circuit.SetActiveBus(bus)
            kv_base[bus] = float(api.Bus.kVBase())
        # Provisional label; final labels need the elements (split phase): conductor_labels().
        nodes.append(Node(idx=idx, bus=bus, conductor="?", opendss_node=name,
                          kv_base_ln=kv_base[bus]))  # fmt: skip
    return nodes, {name: i for i, name in enumerate(names)}


def _terminals(raw: _Raw, index: dict[str, int]) -> list[list[int]]:
    terminals = []
    for t in range(raw.n_terminals):
        positions = raw.node_order[t * raw.n_conductors : (t + 1) * raw.n_conductors]
        nodes = []
        for k in positions:
            if k == 0:
                nodes.append(GROUND)
                continue
            name = f"{raw.buses[t]}.{k}"
            if name not in index:
                raise ExtractionError(f"{raw.name}: node {name} is not in YNodeOrder")
            nodes.append(index[name])
        terminals.append(nodes)
    return terminals


def _controlled(ctx: DssContext) -> set[str]:
    """Full names (lower case) of the elements whose Yprim depends on the network state."""
    api = ctx.api
    names = set()
    i = api.RegControls.First()
    while i > 0:
        names.add(f"transformer.{api.RegControls.Transformer().lower()}")
        i = api.RegControls.Next()
    i = api.CapControls.First()
    while i > 0:
        names.add(f"capacitor.{api.CapControls.Capacitor().lower()}")
        i = api.CapControls.Next()
    return names


def _ratings(ctx: DssContext, raw: _Raw, element_class: str) -> tuple[float | None, float | None]:
    api = ctx.api
    if element_class in ("line", "switch"):
        ctx.activate_element(raw.name)
        return float(api.CktElement.NormalAmps()), None
    if element_class == "transformer":
        api.Transformers.Name(raw.name.split(".", 1)[1])
        api.Transformers.Wdg(1)
        return None, float(api.Transformers.kVA()) * 1e3
    return None, None


def extract_electrical(ctx: DssContext, network_id: str, variant_id: str) -> ElectricalModel:
    """Read the compiled circuit of ``ctx`` into an :class:`ElectricalModel`.

    Steps: nodes from ``YNodeOrder`` (labels a, b, c, n or s1, s2, n in
    split-phase zones, :func:`conductor_labels`; LN base from
    ``Bus.kVBase``), elements from ``AllElementNames`` with terminal maps from ``NodeOrder``,
    Yprims of enabled PD elements, tap dependence (RegControl transformers, CapControl
    capacitors), grounded neutrals (admittance to ground >= ``GROUNDING_MIN_S``, from the row
    sums of the network admittance), source nodes (Vsource terminal 1). Controllers and
    meters are skipped; disabled elements are left out (their terminal maps are not readable,
    see :func:`disabled_elements`); enabled elements of other classes raise
    :class:`UnsupportedElementError`.
    """
    api = ctx.api
    nodes, index = _nodes(ctx)
    raws, disabled = _raw_elements(ctx)
    supported = PD_CLASSES | PC_CLASSES | SOURCE_CLASSES | CONTROL_CLASSES
    unknown = sorted({r.element_class for r in raws} - supported)
    if unknown:
        raise UnsupportedElementError([r.name for r in raws if r.element_class in unknown])
    if disabled:
        log.info("left out %d disabled elements: %s", len(disabled), [n for n, _ in disabled][:10])
    kept = [r for r in raws if r.element_class in PD_CLASSES | PC_CLASSES | SOURCE_CLASSES]
    log.debug("skipped %d control/meter elements", len(raws) - len(kept))

    # Cross-check the PD classification with the engine: PDElements.AllNames() lists all PD
    # elements, disabled ones included; First/NextPDElement only the enabled ones.
    pd_names = {r.name.lower() for r in kept if r.element_class in PD_CLASSES}
    disabled_names = {name.lower() for name, _ in disabled}
    engine_pd = {str(n).lower() for n in api.PDElements.AllNames()} - disabled_names
    enabled_engine_pd = _enabled_pd_names(ctx)
    if engine_pd != pd_names or enabled_engine_pd != pd_names:
        diff = sorted((engine_pd ^ pd_names) | (enabled_engine_pd ^ pd_names))
        raise ExtractionError(f"PD classification differs from the engine's PD lists: {diff[:10]}")

    controlled = _controlled(ctx)
    elements: list[Element] = []
    source_nodes: set[int] = set()
    for idx, raw in enumerate(kept):
        is_pd = raw.element_class in PD_CLASSES
        element_class = raw.element_class
        if element_class == "line":
            api.Lines.Name(raw.name.split(".", 1)[1])
            if api.Lines.IsSwitch():
                element_class = "switch"
        terminals = _terminals(raw, index)
        yprim = ctx.element_yprim(raw.name) if is_pd else None
        rating_a, rating_va = _ratings(ctx, raw, element_class) if is_pd else (None, None)
        elements.append(
            Element(
                idx=idx,
                name=raw.name,
                element_class=element_class,
                is_pd=is_pd,
                enabled=True,
                terminals=terminals,
                yprim=yprim,
                is_tap_dependent=raw.name.lower() in controlled,
                rating_a=rating_a,
                rating_va=rating_va,
            )
        )
        if element_class == "vsource":
            source_nodes.update(n for n in terminals[0] if n != GROUND)

    try:
        labels = conductor_labels([n.opendss_node for n in nodes], elements)
    except PhaseError as exc:
        raise ExtractionError(str(exc)) from exc
    nodes = [replace(n, conductor=label) for n, label in zip(nodes, labels, strict=True)]

    # Neutral grounding, whatever element provides it (reactor to ground, transformer Rneut/Xneut,
    # any shunt): the row sum of Y_state at the neutral node.
    draft = ElectricalModel(network_id=network_id, variant_id=variant_id, nodes=nodes,
                            elements=elements)  # fmt: skip
    y_ground = draft.ground_admittance()
    grounding = {
        n.idx: 1.0 / complex(y_ground[n.idx]) + 0j  # + 0j: no negative zeros
        for n in nodes
        if n.conductor == "n" and abs(y_ground[n.idx]) >= GROUNDING_MIN_S
    }
    nodes = [
        Node(
            idx=n.idx,
            bus=n.bus,
            conductor=n.conductor,
            opendss_node=n.opendss_node,
            kv_base_ln=n.kv_base_ln,
            is_grounded=n.idx in grounding,
            r_ground_ohm=grounding[n.idx].real if n.idx in grounding else None,
            x_ground_ohm=grounding[n.idx].imag if n.idx in grounding else None,
            is_source=n.idx in source_nodes,
        )
        for n in nodes
    ]
    if not source_nodes:
        raise ExtractionError("no enabled Vsource: the model has no slack nodes")
    return ElectricalModel(network_id=network_id, variant_id=variant_id, nodes=nodes,
                           elements=elements)  # fmt: skip
