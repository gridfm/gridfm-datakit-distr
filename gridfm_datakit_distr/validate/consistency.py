"""Cross-check of the network admittance against the engine's own system Y (check C3).

The OpenDSS solver uses ``Y_solver = Y_state + Σ Yprim(PC elements) + Σ Yprim(sources)``
(measured: equal to 1.9e-16 entrywise on IEEE 13). Comparing the model's ``Y_state`` with
``Y_solver`` minus the PC and source part checks the extraction end to end: node order,
terminal maps, Yprim layout and the PD/PC split.

Each entry's difference is measured against the largest magnitude entering the comparison
(``|Y_state|``, ``|Y_solver|``, ``|Y_pc+sources|``): where a load is the only element coupling
two conductors, ``Y_solver - Y_pc`` cancels to round-off (at most 5.7e-17 S on EDP)
and the entry is judged at the scale of the cancelled terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from gridfm_datakit_distr.model.electrical import ElectricalModel


@dataclass(frozen=True, slots=True)
class AdmittanceReport:
    """Entrywise comparison of two admittance matrices on the union of their non-zeros."""

    max_rel: float
    """max |a_ij - b_ij| / max(|a_ij|, |y_solver_ij|, |y_pc_ij|)."""
    worst: tuple[str, str]
    """OpenDSS node names (row, column) of the worst relative entry."""
    n_entries: int

    def passes(self, rel_tol: float = 1e-9) -> bool:
        """The C3 criterion."""
        return self.max_rel <= rel_tol


def compare_admittance(
    model: ElectricalModel,
    y_state: sp.csr_matrix,
    y_solver: sp.csr_matrix,
    y_pc_sources: sp.csr_matrix,
) -> AdmittanceReport:
    """Compare ``y_state`` with ``y_solver - y_pc_sources`` entry by entry."""
    n = model.n_nodes
    for name, m in (("y_state", y_state), ("y_solver", y_solver), ("y_pc_sources", y_pc_sources)):
        if m.shape != (n, n):
            raise ValueError(f"{name} has shape {m.shape}, the model has {n} nodes")
    a = sp.csr_matrix(y_state)
    b = sp.csr_matrix(y_solver - y_pc_sources)
    pattern = sp.csr_matrix((abs(a) + abs(b)) > 0)
    rows, cols = pattern.nonzero()
    if rows.size == 0:
        return AdmittanceReport(0.0, ("", ""), 0)
    av = np.asarray(a[rows, cols]).ravel()
    bv = np.asarray(b[rows, cols]).ravel()
    solver = np.abs(np.asarray(sp.csr_matrix(y_solver)[rows, cols]).ravel())
    pc = np.abs(np.asarray(sp.csr_matrix(y_pc_sources)[rows, cols]).ravel())
    diff = np.abs(av - bv)
    rel = diff / np.maximum.reduce([np.abs(av), solver, pc])
    k = int(np.argmax(rel))
    names = [node.opendss_node for node in model.nodes]
    return AdmittanceReport(
        max_rel=float(rel[k]),
        worst=(names[int(rows[k])], names[int(cols[k])]),
        n_entries=int(rows.size),
    )
