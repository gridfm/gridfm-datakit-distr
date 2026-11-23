"""Which networks the OPF problem covers, and why the others are left out.

The problem of :mod:`gridfm_datakit_distr.opf.problem` covers networks whose power-delivery
elements (lines, switches, transformers, capacitors, reactors) enter through their admittances at
the network state, with one voltage source grounded at its second terminal, loads, PV units
(``Generator.pv_<load>``) and storage units (``Generator.bess_<load>``, fixed at their set-point)
connected wye (phases to a neutral node or to ground) or delta, on any number of phases
(:func:`~gridfm_datakit_distr.model.ground.unit_branches`). Other generators, PV systems and
``Storage`` elements are not covered yet. Also the OPF's bounds evaluated at given voltages
(:func:`vpn_bounds`, :func:`winding_va`), usable without the OPF. No CasADi here: the check runs
without the ``opf`` extra.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.ground import (
    conventional_zones,
    floating_zones,
    is_conventional,
    unit_branches,
)

PV_PREFIX = "generator.pv_"
"""PV units are the generators the variants place, ``Generator.pv_<load>``."""
BESS_PREFIX = "generator.bess_"
"""Storage units, ``Generator.bess_<load>``: fixed injections in the OPF."""
FloatArray = NDArray[np.float64]
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
    if cls == "generator" and not element.name.lower().startswith((PV_PREFIX, BESS_PREFIX)):
        return "generator that is not a PV or storage unit"
    if cls in ("load", "generator"):
        covered = unit_branches(model, element) is not None
        return None if covered else f"{cls} with a connection not covered by the OPF"
    return f"{cls} not covered by the OPF yet"


def vpn_bounds(
    model: ElectricalModel, vmin_pu: float, vmax_pu: float
) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    """:func:`phase_neutral_pairs` and the bounds (V) of their voltage magnitudes: ``vmin_pu``
    and ``vmax_pu`` times the pair's nominal voltage (:func:`pair_nominal_v`)."""
    pairs = phase_neutral_pairs(model)
    nominal = pair_nominal_v(model, pairs)
    return pairs, vmin_pu * nominal, vmax_pu * nominal


def pair_nominal_v(model: ElectricalModel, pairs: NDArray[np.int64]) -> FloatArray:
    """Nominal voltage (V) of each pair: the phase node's nominal line-to-neutral voltage, times
    √3 between two phases."""
    ln = np.array([model.nodes[int(i)].nominal_v_ln for i in pairs[:, 0]], dtype=np.float64)
    two_phases = np.array(
        [j != GROUND and model.nodes[int(j)].conductor != "n" for j in pairs[:, 1]], dtype=bool
    )
    return np.where(two_phases, np.sqrt(3.0) * ln, ln)


def phase_neutral_pairs(model: ElectricalModel) -> NDArray[np.int64]:
    """``(k, 2)``: each phase node and its bus's neutral node (:data:`GROUND` if none). At a bus
    without neutral in a zone whose voltage to ground is a numerical convention
    (:func:`conventional_zones`), the bus's phase-to-phase pairs instead of pairs to ground:
    ``(a, b), (b, c), (c, a)``, or the one pair of two phases."""
    conventional = {i for z in conventional_zones(model) for i in z}
    by_bus: dict[str, dict[str, int]] = {}
    for node in model.nodes:
        by_bus.setdefault(node.bus, {})[node.conductor] = node.idx
    pairs: list[tuple[int, int]] = []
    for nodes in by_bus.values():
        phases = [idx for conductor, idx in nodes.items() if conductor != "n"]
        if phases and "n" not in nodes and phases[0] in conventional:
            ring = list(zip(phases, phases[1:] + phases[:1], strict=True))
            pairs += ring if len(phases) > 2 else ring[:1] if len(phases) == 2 else []
        else:
            pairs += [(idx, nodes.get("n", GROUND)) for idx in phases]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def winding_va(element: Element, v: NDArray[np.complex128]) -> float:
    """``|S|`` (VA) into winding 1 of a transformer at node voltages ``v`` (V), from its Yprim."""
    assert element.yprim is not None
    local = np.array([0.0 if n == GROUND else v[n] for n in element.local_nodes()])
    first = element.n_conductors
    current = element.yprim[:first] @ local
    return float(abs(np.sum(local[:first] * np.conj(current))))
