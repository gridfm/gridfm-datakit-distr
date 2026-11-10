"""Currents and admittances of power-conversion elements and sources, read from the engine.

Used by the residual check and the admittance cross-check C3. Currents
reported by OpenDSS flow *into* the element terminals; the injection into the network is their
opposite.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext, complex_from_interleaved
from gridfm_datakit_distr.extract.classes import SOURCE_CLASSES
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element


def _is_source(element: Element) -> bool:
    return element.element_class in SOURCE_CLASSES


def injections(
    ctx: DssContext, model: ElectricalModel, *, sources: bool = False
) -> NDArray[np.complex128]:
    """Net current (A) injected into each node by the PC elements (or by the sources).

    With ``sources=False`` the loads, generators, PV and storage; with ``sources=True`` the
    Vsources only. The model must come from the same compiled circuit.
    """
    out = np.zeros(model.n_nodes, dtype=np.complex128)
    for element in model.elements:
        if element.is_pd or _is_source(element) != sources:
            continue
        ctx.activate_element(element.name)
        with ctx.engine_call(f"CktElement.Currents {element.name}"):
            currents = complex_from_interleaved(ctx.api.CktElement.Currents())
        for node, current in zip(element.local_nodes(), currents, strict=True):
            if node != GROUND:
                out[node] -= current
    return out


def pc_source_admittance(ctx: DssContext, model: ElectricalModel) -> sp.csr_matrix:
    """Σ of the Yprims (S) of PC elements and sources, at node level.

    Together with the network admittance this is the system Y the solver uses.
    """
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    vals: list[NDArray[np.complex128]] = []
    for element in model.elements:
        if element.is_pd:
            continue
        yprim = ctx.element_yprim(element.name)
        local = np.array(element.local_nodes(), dtype=np.int64)
        keep = np.flatnonzero(local != GROUND)
        idx = local[keep]
        rows.append(np.repeat(idx, idx.size))
        cols.append(np.tile(idx, idx.size))
        vals.append(yprim[np.ix_(keep, keep)].ravel())
    if not vals:
        empty = np.zeros(0, dtype=np.int64)
        return sp.csr_matrix(
            (np.zeros(0, dtype=np.complex128), (empty, empty)), shape=(model.n_nodes,) * 2
        )
    return sp.csr_matrix(
        sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(model.n_nodes, model.n_nodes),
        )
    )
