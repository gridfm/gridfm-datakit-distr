"""Vectorised result extraction.

For the enabled power-delivery elements of a model, every local position (terminal x conductor)
gets a row: ``S`` (positions x nodes) selects its node voltage (ground positions: 0 V) and the
block-diagonal ``Yblk`` of the Yprims gives the currents flowing into the elements,

    I_local = Yblk (S V),        I_inj = Sᵀ I_local = Y_state V,        S_inj = V ∘ conj(I_inj),

one sparse product per scenario. ``I_inj`` is the current that power-conversion elements and sources
inject into the network at each node (not stored: SCHEMA ``node_results``). Tap-dependent
elements have their blocks in a separate matrix rebuilt per network state
(:meth:`ExtractionPlan.at_state`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gridfm_datakit_distr.model.electrical import ElectricalModel, Element


@dataclass(frozen=True, slots=True, eq=False)
class ExtractionPlan:
    element_idx: NDArray[np.int32]
    """Element of each local position (row)."""
    terminal: NDArray[np.int8]
    position: NDArray[np.int8]
    """Conductor position within the terminal."""
    select: sp.csr_matrix
    """``S``: positions x nodes."""
    fixed: sp.csr_matrix
    """Blocks of the tap-independent elements."""
    dependent: sp.csr_matrix
    """Blocks of the tap-dependent elements at the plan's state."""
    offsets: dict[int, int]
    """Element index -> first row of its block."""

    @property
    def yblk(self) -> sp.csr_matrix:
        return (self.fixed + self.dependent).tocsr()

    def currents(self, v: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Currents into the elements at each local position (A)."""
        return np.asarray(self.fixed @ (self.select @ v) + self.dependent @ (self.select @ v))

    def injections(self, i_local: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Net current injected into the network at each node (A): ``Sᵀ I_local``."""
        return np.asarray(self.select.T @ i_local)

    def at_state(self, model: ElectricalModel) -> ExtractionPlan:
        """The plan with the tap-dependent blocks of ``model`` (a model at a network state)."""
        return ExtractionPlan(
            self.element_idx, self.terminal, self.position, self.select, self.fixed,
            _blocks(model, self.offsets, len(self.element_idx), dependent=True), self.offsets,
        )  # fmt: skip


def build_plan(model: ElectricalModel) -> ExtractionPlan:
    """The extraction plan of a model (its tap-dependent elements at the model's state); its rows
    are those of the model's selection matrix."""
    rows_element, rows_terminal, rows_position = [], [], []
    offsets: dict[int, int] = {}
    row = 0
    for e in model.pd_elements():
        offsets[e.idx] = row
        for t in range(e.n_terminals):
            for p in range(e.n_conductors):
                rows_element.append(e.idx)
                rows_terminal.append(t)
                rows_position.append(p)
        row += e.n_positions
    select = model.selection().astype(np.complex128)
    return ExtractionPlan(
        element_idx=np.array(rows_element, dtype=np.int32),
        terminal=np.array(rows_terminal, dtype=np.int8),
        position=np.array(rows_position, dtype=np.int8),
        select=select,
        fixed=_blocks(model, offsets, row, dependent=False),
        dependent=_blocks(model, offsets, row, dependent=True),
        offsets=offsets,
    )


def _blocks(
    model: ElectricalModel, offsets: dict[int, int], n: int, *, dependent: bool
) -> sp.csr_matrix:
    data: list[NDArray[np.complex128]] = []
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    for e in model.pd_elements():
        if e.is_tap_dependent != dependent:
            continue
        y = _yprim(e)
        k = y.shape[0]
        r, c = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        rows.append(offsets[e.idx] + r.ravel())
        cols.append(offsets[e.idx] + c.ravel())
        data.append(y.ravel())
    if not data:
        none = np.zeros(0, dtype=np.int64)
        return sp.csr_matrix((np.zeros(0, dtype=np.complex128), (none, none)), shape=(n, n))
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )


def _yprim(e: Element) -> NDArray[np.complex128]:
    if e.yprim is None:
        raise ValueError(f"{e.name}: an enabled PD element without Yprim")
    return e.yprim
