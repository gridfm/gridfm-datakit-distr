"""Power-balance (KCL) residual of a solved operating point.

At every node, the current entering the network from power-conversion elements must equal the
current the network admittance draws: ``r = Y_state @ v - i_pc = 0``. Source (slack) nodes are
not checked: the source injection balances them.

Pass criterion (C2), at every non-source node i:

    |r_i| <= tau * I_scale + tol * g_i,    g_i = sum_j |Y_ij| |v_j|,  I_scale = max(1 A, max |i_pc|)

``tau`` (1e-6) is the physical tolerance of the plan; ``tol * g_i`` is what a solution converged
to the solver tolerance ``tol`` can certify at a node where large currents cancel (stiff
elements). Reported alongside: the plan's ``rel = max|r| / I_scale`` and the backward error
``max_i |r_i| / (g_i + |i_pc,i|)`` over the nodes whose gross current ``g_i + |i_pc,i|`` reaches
``tau * I_scale``: below that floor (neutrals carrying almost no current) a relative error is
round-off noise, and those nodes are covered by the ``tau * I_scale`` term of C2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gridfm_datakit_distr.model.electrical import ElectricalModel

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class ResidualReport:
    """KCL residual over the non-source nodes."""

    max_abs_a: float
    """max |r_i| (A)."""
    rel: float
    """max |r| / I_scale, the residual measure."""
    backward: float
    """max_i |r_i| / (g_i + |i_pc,i|) over nodes with g_i + |i_pc,i| >= tau * I_scale."""
    ratio: float
    """max_i |r_i| / (tau * I_scale + tol * g_i): C2 passes iff ratio <= 1."""
    worst_node: str
    """OpenDSS name of the node with the largest ratio."""
    n_checked: int

    def passes(self) -> bool:
        """Check C2."""
        return self.ratio <= 1.0


def residual(
    model: ElectricalModel,
    v: NDArray[np.complex128],
    i_pc: NDArray[np.complex128],
    *,
    tolerance: float,
    tau: float = 1e-6,
    ybus: sp.csr_matrix | None = None,
) -> ResidualReport:
    """KCL residual of ``v`` (V to ground, model node order).

    ``i_pc`` is the net current (A) injected at each node by the power-conversion elements
    (loads, generators, PV, storage), sources excluded. ``tolerance`` is the solver tolerance
    the solution was obtained with; ``ybus`` defaults to ``model.ybus()``.
    """
    if not tolerance >= 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    y = model.ybus() if ybus is None else ybus
    n = model.n_nodes
    if v.shape != (n,) or i_pc.shape != (n,) or y.shape != (n, n):
        raise ValueError(f"shapes do not match the model ({n} nodes)")
    r = y @ v - i_pc
    checked = np.ones(n, dtype=bool)
    checked[model.source_nodes()] = False
    idx = np.flatnonzero(checked)
    if idx.size == 0:
        raise ValueError("the model has no non-source node to check")
    g = abs(y) @ np.abs(v)
    abs_r = np.abs(r)
    i_scale = max(1.0, float(np.abs(i_pc[idx]).max()))
    bound = tau * i_scale + tolerance * g
    gross = g + np.abs(i_pc)
    active = idx[gross[idx] >= tau * i_scale]
    backward = float((abs_r[active] / gross[active]).max()) if active.size else 0.0
    ratio = abs_r[idx] / bound[idx]
    worst = int(idx[np.argmax(ratio)])
    return ResidualReport(
        max_abs_a=float(abs_r[idx].max()),
        rel=float(abs_r[idx].max()) / i_scale,
        backward=backward,
        ratio=float(ratio.max()),
        worst_node=model.nodes[worst].opendss_node,
        n_checked=int(idx.size),
    )


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def check_this_scenario(scenario_id: int, rate: float) -> bool:
    """Whether a production run checks the residual of this scenario (``residual_sample_rate``).

    A deterministic draw of probability ``rate`` from a hash of the scenario id: the same
    scenarios are checked whatever the execution order or the number of workers.
    """
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return _splitmix64(scenario_id & _MASK64) < int(rate * 2**64)
