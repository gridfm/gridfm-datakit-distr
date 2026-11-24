"""Power flow from the stored tables alone.

The network is ``A V = Y_s E + I(V)`` with ``A = Y_state + Y_s`` (the source admittance embedded at
its nodes), ``E`` the scenario's ideal source voltages (``ScenarioData.source_e``) and ``I(V)`` the
currents the constant-power loads and DER inject: each branch ``(a, b)`` of a unit draws
``i = conj(s / (V_a - V_b))`` out of ``a`` and into ``b``, ``s`` an equal share of the unit's
power. :func:`fixed_point` iterates
``V ← A⁻¹ (Y_s E + I(V))`` (the Z-bus method) from the no-load voltage. Errors are per unit of
each node's line-to-neutral base. Engine-free, like the reader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from gridfm_datakit_distr.io.reader import ScenarioData
from gridfm_datakit_distr.model.electrical import GROUND

TOL_PU = 1e-13
"""Fixed point: stop when no node moves more than this (pu) in one iteration..."""
FLOOR_PU = 1e-8
"""...or when the step, below this, stops shrinking: round-off of the solve (IEEE 13, whose 1e7 S
switch puts the per-unit matrix's condition number at 2e8, levels off near 4e-10 pu)."""
MAX_ITER = 500


@dataclass(frozen=True, slots=True)
class ClosedLoop:
    scenario_id: int
    iterations: int
    max_dv_pu: float
    """Largest ``|V_fixed_point - V_stored|`` per unit of the node's base."""


class _System:
    """The factorised ``A`` and the source's right-hand side of one scenario."""

    def __init__(self, sc: ScenarioData) -> None:
        e = sc.source_e
        if sc.y_state is None or e is None:
            raise ValueError(f"scenario {sc.scenario_id} has no state or no source voltage")
        var = sc.variant
        n, k = var.model.n_nodes, var.source_nodes.size
        where = (np.repeat(var.source_nodes, k), np.tile(var.source_nodes, k))
        ys = sp.coo_matrix((var.y_s.ravel(), where), shape=(n, n))
        self.d = sp.diags(var.base_v)  # solved per unit of the node bases: better conditioned
        self.lu = spla.splu(sp.csc_matrix(self.d @ (sc.y_state + ys) @ self.d))
        self.rhs = np.zeros(n, dtype=np.complex128)
        self.rhs[var.source_nodes] = var.y_s @ e
        a, b, s = [], [], []
        for idx, power in sc.s_drawn.items():
            br = var.branches[idx]
            for x, y in br:
                a.append(x)
                b.append(y)
                s.append(power / len(br))
        self.a, self.b = np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)
        self.s = np.array(s, dtype=np.complex128)
        self.n = n

    def injections(self, v: NDArray[np.complex128]) -> NDArray[np.complex128]:
        grounded = self.b == GROUND
        vb = np.where(grounded, 0.0, v[np.where(grounded, 0, self.b)])
        i = np.conj(self.s / (v[self.a] - vb))
        out = np.zeros(self.n, dtype=np.complex128)
        np.subtract.at(out, self.a, i)
        np.add.at(out, self.b[~grounded], i[~grounded])
        return out

    def solve(self, injections: NDArray[np.complex128]) -> NDArray[np.complex128]:
        v_pu = self.lu.solve(self.d @ (self.rhs + injections))
        return np.asarray(self.d @ v_pu, dtype=np.complex128)


def fixed_point(sc: ScenarioData, max_iter: int = MAX_ITER) -> tuple[NDArray[np.complex128], int]:
    """Node voltages (V) of the power flow and the iterations it took; ``ValueError`` if it does
    not settle within ``max_iter``."""
    system = _System(sc)
    base = sc.variant.base_v
    v = system.solve(np.zeros(system.n, dtype=np.complex128))
    previous = np.inf
    for k in range(1, max_iter + 1):
        new = system.solve(system.injections(v))
        step = float(np.max(np.abs(new - v) / base))
        v = new
        if step <= TOL_PU or FLOOR_PU > step >= previous:
            return v, k
        previous = step
    raise ValueError(f"scenario {sc.scenario_id}: no fixed point in {max_iter} iterations")


def closed_loop(sc: ScenarioData) -> ClosedLoop:
    """The fixed point of ``sc`` against its stored voltages."""
    if sc.v is None:
        raise ValueError(f"scenario {sc.scenario_id} has no stored voltages")
    v, iterations = fixed_point(sc)
    err = float(np.max(np.abs(v - sc.v) / sc.variant.base_v))
    return ClosedLoop(scenario_id=sc.scenario_id, iterations=iterations, max_dv_pu=err)
