"""Electrical layer of the internal model: nodes, elements and primitive admittances.

Pure data and linear algebra, no OpenDSS. Conventions (outputs.md §1, §4):

- node indices are dense and 0-based; ground is never a node: a terminal position tied to
  ground has node index :data:`GROUND` (-1);
- an element's *local ordering* is terminal-major (terminal 0, positions 0..n_cond-1, then
  terminal 1, ...), the order of OpenDSS ``CktElement.NodeOrder``;
- ``yprim`` is the primitive admittance matrix in S, in local ordering, so that the currents
  flowing into the element are ``yprim @ v_local`` (ground positions have v = 0);
- ``kv_base_ln`` is the line-to-neutral base voltage of the node's bus, in kV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

GROUND = -1

ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class Node:
    """One node (a conductor of a bus)."""

    idx: int
    bus: str
    conductor: str
    """``a``, ``b``, ``c``, ``n``, ``s1``, ``s2`` (outputs.md §1)."""
    opendss_node: str
    """OpenDSS name ``bus.k``."""
    kv_base_ln: float
    is_grounded: bool = False
    r_ground_ohm: float | None = None
    x_ground_ohm: float | None = None
    is_source: bool = False
    kv_nominal_ln: float | None = None
    """Nominal line-to-neutral voltage of the bus (kV) where the catalog declares one different
    from the base; None: the base. Voltage limits are per unit of it."""

    @property
    def nominal_v_ln(self) -> float:
        """Nominal line-to-neutral voltage (V): :attr:`kv_nominal_ln`, else the base."""
        kv = self.kv_base_ln if self.kv_nominal_ln is None else self.kv_nominal_ln
        return kv * 1e3


@dataclass(eq=False, slots=True)
class Element:
    """A circuit element: its terminal map and, for PD elements, its primitive admittance."""

    idx: int
    name: str
    """OpenDSS full name, e.g. ``Line.650632``."""
    element_class: str
    """Lower-case class: ``line``, ``transformer``, ``load``, ... (outputs.md §4)."""
    is_pd: bool
    enabled: bool
    terminals: list[list[int]]
    """Node index per terminal and conductor position; :data:`GROUND` for ground."""
    yprim: ComplexMatrix | None = None
    """Primitive admittance (S), local ordering; ``None`` for power-conversion elements."""
    is_tap_dependent: bool = False
    rating_a: float | None = None
    """Normal ampacity per conductor (A); stored as float32."""
    rating_va: float | None = None
    """Apparent-power rating (VA); stored as float32."""

    def __post_init__(self) -> None:
        if not self.terminals or not self.terminals[0]:
            raise ValueError(f"{self.name}: an element needs at least one terminal and conductor")
        n_conductors = len(self.terminals[0])
        if any(len(terminal) != n_conductors for terminal in self.terminals):
            raise ValueError(f"{self.name}: terminals with different numbers of conductors")
        if self.yprim is not None:
            n = self.n_positions
            if self.yprim.shape != (n, n) or self.yprim.dtype != np.complex128:
                raise ValueError(
                    f"{self.name}: yprim must be complex128 ({n}, {n}), "
                    f"got {self.yprim.dtype} {self.yprim.shape}"
                )

    @property
    def n_terminals(self) -> int:
        return len(self.terminals)

    @property
    def n_conductors(self) -> int:
        return len(self.terminals[0])

    @property
    def n_positions(self) -> int:
        """Size of the local ordering: terminals x conductors."""
        return self.n_terminals * self.n_conductors

    def local_nodes(self) -> list[int]:
        """Node index of every local position (terminal-major)."""
        return [node for terminal in self.terminals for node in terminal]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Element):
            return NotImplemented
        if self.yprim is None or other.yprim is None:
            same_yprim = self.yprim is None and other.yprim is None
        else:
            same_yprim = bool(np.array_equal(self.yprim, other.yprim))
        return same_yprim and (
            self.idx,
            self.name,
            self.element_class,
            self.is_pd,
            self.enabled,
            self.terminals,
            self.is_tap_dependent,
            self.rating_a,
            self.rating_va,
        ) == (
            other.idx,
            other.name,
            other.element_class,
            other.is_pd,
            other.enabled,
            other.terminals,
            other.is_tap_dependent,
            other.rating_a,
            other.rating_va,
        )


@dataclass(slots=True)
class ElectricalModel:
    """Nodes and elements of one network variant, at one network state."""

    network_id: str
    variant_id: str
    nodes: list[Node]
    elements: list[Element]

    def __post_init__(self) -> None:
        for i, node in enumerate(self.nodes):
            if node.idx != i:
                raise ValueError(f"node {node.opendss_node!r} has index {node.idx}, expected {i}")
        for i, element in enumerate(self.elements):
            if element.idx != i:
                raise ValueError(f"element {element.name!r} has index {element.idx}, expected {i}")
        n = self.n_nodes
        for element in self.elements:
            bad = [node for node in element.local_nodes() if not GROUND <= node < n]
            if bad:
                raise ValueError(f"{element.name}: node indices out of range: {bad}")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def source_nodes(self) -> NDArray[np.int64]:
        """Indices of the source (slack) nodes."""
        return np.array([node.idx for node in self.nodes if node.is_source], dtype=np.int64)

    def pd_elements(self) -> list[Element]:
        """Enabled power-delivery elements, in element order: those of the network admittance."""
        elements = [e for e in self.elements if e.is_pd and e.enabled]
        missing = [e.name for e in elements if e.yprim is None]
        if missing:
            raise ValueError(f"enabled PD elements without Yprim: {missing[:5]}")
        return elements

    def selection(self) -> sp.csr_matrix:
        """``S``, local positions ← nodes: rows follow the local orderings of
        :meth:`pd_elements` one after the other (the rows of the element results); ground
        positions have zero rows."""
        rows: list[int] = []
        cols: list[int] = []
        offset = 0
        for element in self.pd_elements():
            for k, node in enumerate(element.local_nodes()):
                if node != GROUND:
                    rows.append(offset + k)
                    cols.append(node)
            offset += element.n_positions
        return sp.csr_matrix(
            (np.ones(len(rows)), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(offset, self.n_nodes),
        )

    def element_blocks(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``(Yblk, S)``: block diagonal of the PD Yprims and :meth:`selection`. Element currents
        are ``Yblk @ (S @ v)`` and the network admittance is ``Sᵀ Yblk S``."""
        elements = self.pd_elements()
        selection = self.selection()
        if not elements:
            return sp.csr_matrix(np.zeros((0, 0), dtype=np.complex128)), selection
        # one CSR from every block's non-zeros (row-major, as a COO of each block would list them)
        rows, cols, values = [], [], []
        offset = 0
        for element in elements:
            y = element.yprim
            assert y is not None  # pd_elements checked it
            r, c = np.nonzero(y)
            rows.append(r + offset)
            cols.append(c + offset)
            values.append(y[r, c])
            offset += y.shape[0]
        yblk = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(offset, offset),
        )
        return yblk, selection

    def ground_admittance(self) -> NDArray[np.complex128]:
        """Admittance (S) from each node to ground through the PD elements: row sums of Y_state.

        Series branches contribute nothing to a row sum; shunts do: grounding impedances
        (reactors to ground, transformer ``Rneut``/``Xneut``), capacitors, line charging and the
        transformers' anti-float admittance.
        """
        return np.asarray(self.ybus().sum(axis=1), dtype=np.complex128).ravel()

    def ybus(self) -> sp.csr_matrix:
        """Network admittance ``Y_state`` (S): Σ over enabled PD elements of ``Sᵀ Yprim S``.

        Includes lines, transformers, switches, reactors and capacitors at the current state;
        excludes power-conversion elements and the source impedance.
        """
        yblk, selection = self.element_blocks()
        return sp.csr_matrix(selection.T @ yblk @ selection)
