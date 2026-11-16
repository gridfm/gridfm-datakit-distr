"""The OPF problem of one network state: a nodal current-voltage NLP solved with Ipopt (CasADi).

Unknowns: the node voltages in rectangular coordinates, one current per load and PV unit, and the
active and reactive power of each PV unit. Constraints:

- Kirchhoff's current law at every node, ``(Y_state + Y_s) V - Y_s E = C_g I_g - C_l I_l``,
  linear, entered as one constant sparse matrix: ``Y_state`` is :meth:`ElectricalModel.ybus`,
  ``Y_s`` and ``E`` the source's admittance and ideal voltages (:class:`Source`);
- each device between its two nodes ``i, j``: ``(V_i - V_j) conj(I) = P + jQ``, the load's
  constant power, the PV unit's ``p + jq``;
- PV units: ``0 <= p <= p_avail``, ``p^2 + q^2 <= s_rated^2``;
- ``v_min <= |V_p - V_n| <= v_max`` at every phase node of every bus (``|V_p|`` where the bus
  has no neutral node).

Objective: ``sum_k (c2_k p_k^2 + c1_k p_k)`` over the source and the PV units (:class:`Costs`),
``p_k`` the active power element ``k`` delivers; the source's is that of its ideal voltages,
``Re sum E conj(Y_s (E - V_s))``. The model is built once per network state; the source voltage
and the costs are CasADi parameters, the loads, PV limits and ratings enter as the bounds of the
call, so each scenario is one Ipopt solve from a flat start. The NLP is scaled to per unit on the
source's line-to-neutral base voltage and :data:`S_BASE_VA`; inputs and results are SI.

Requires the ``opf`` extra (CasADi).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.opf.support import unsupported_reason

S_BASE_VA = 1e3
"""Power base of the NLP (VA). shortcut: the scale tested on LV feeders; revisit for MV networks."""
IPOPT_OPTIONS: dict[str, Any] = {"tol": 1e-9, "max_iter": 3000}
"""With ``max_wall_time`` from :class:`OPFProblem` (a limit hit ends as
``Maximum_WallTime_Exceeded``)."""
SOLVED = "Solve_Succeeded"
"""The only Ipopt status that counts as solved."""

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Source:
    """The voltage source: ideal voltages behind an admittance, at its terminal's nodes."""

    nodes: NDArray[np.int64]
    """Model nodes of the source terminal, in conductor order."""
    y_s: NDArray[np.complex128]
    """Admittance (S) between the ideal source and :attr:`nodes`."""
    e_v: NDArray[np.complex128]
    """Ideal source voltages (V) at 1 pu; a scenario's are ``source_v_pu * e_v``."""


def read_source(ctx: DssContext, model: ElectricalModel) -> Source:
    """The source of ``model`` from its compiled circuit in ``ctx``.

    :class:`ElectricalModel` keeps neither the Vsource's Yprim (it is not a PD element) nor its
    base voltage and angle, so they are read from the engine: ``Y_s`` is the Yprim block of the
    first terminal (the second is ground, :func:`unsupported_reason`), ``E`` has the magnitude
    ``BasekV`` (line-to-neutral) and the angles ``AngleDeg - 120°·k`` per conductor ``k``.
    """
    (vs,) = [e for e in model.elements if e.enabled and e.element_class == "vsource"]
    keep = [k for k, node in enumerate(vs.terminals[0]) if node != GROUND]
    y_s = ctx.element_yprim(vs.name)[np.ix_(keep, keep)]
    api = ctx.api
    with ctx.engine_call(f"Vsources {vs.name}"):
        api.Vsources.Name(vs.name.split(".", 1)[1])
        base_kv, angle_deg = float(api.Vsources.BasekV()), float(api.Vsources.AngleDeg())
        phases = int(api.Vsources.Phases())
    v_ln = base_kv * 1e3 / (np.sqrt(3.0) if phases > 1 else 1.0)
    angles = np.deg2rad(angle_deg) - 2.0 * np.pi / 3.0 * np.array(keep)
    nodes = np.array([vs.terminals[0][k] for k in keep], dtype=np.int64)
    return Source(nodes=nodes, y_s=y_s, e_v=v_ln * np.exp(1j * angles))


@dataclass(frozen=True, slots=True, eq=False)
class Costs:
    """Coefficients of ``c2 p^2 + c1 p`` per controllable element: the source, then the PV units
    in :attr:`OPFProblem.pv_names` order; ``p`` in W, ``c1`` per W, ``c2`` per W^2."""

    c1: FloatArray
    c2: FloatArray


def economic(n_pv: int) -> Costs:
    """Preset ``economic``: source ``c1 = 1``, everything else 0. The objective is then the power
    the source delivers (W): the minimum import."""
    c1 = np.zeros(1 + n_pv)
    c1[0] = 1.0
    return Costs(c1=c1, c2=np.zeros(1 + n_pv))


@dataclass(frozen=True, slots=True, eq=False)
class OPFResult:
    """One solve, whatever its outcome."""

    status: str
    """Ipopt's return status as returned; only :data:`SOLVED` counts as solved."""
    objective: float
    """``sum c2 p^2 + c1 p`` at the returned point (cost units; W with :func:`economic`)."""
    iterations: int
    t_solve_s: float
    v: NDArray[np.complex128]
    """Node voltages to ground (V), in the model's node order."""
    pv_p_w: FloatArray
    """Active power each PV unit injects (W), in :attr:`OPFProblem.pv_names` order."""
    pv_q_var: FloatArray
    """Reactive power each PV unit injects (var)."""

    @property
    def solved(self) -> bool:
        return self.status == SOLVED


class OPFProblem:
    """The NLP of one network state, built once; each scenario is one :meth:`solve`."""

    def __init__(
        self,
        model: ElectricalModel,
        source: Source,
        vmin_pu: float,
        vmax_pu: float,
        time_limit_s: float = 300.0,
    ) -> None:
        """``vmin_pu``, ``vmax_pu``: bounds of the phase-to-neutral voltage magnitudes, per unit
        of each phase node's nominal line-to-neutral voltage (:func:`vpn_bounds`);
        ``time_limit_s``: wall-clock limit of one solve (s)."""
        self.pairs, self.vpn_lo_v, self.vpn_hi_v = vpn_bounds(model, vmin_pu, vmax_pu)
        """Phase and neutral node of every bounded voltage, and its bounds (V)."""
        reason = unsupported_reason(model)
        if reason is not None:
            raise ValueError(f"{model.variant_id} is outside the OPF slice: {reason}")
        loads = _devices(model, "load")
        pvs = _devices(model, "generator")
        self.load_names = tuple(e.name for e in loads)
        """Load elements by name: the order of the scenario's customers."""
        self.pv_names = tuple(e.name for e in pvs)
        """PV generators by name: the order of the variant's PV units."""
        n, nl, ng = model.n_nodes, len(loads), len(pvs)
        self._n, self._nl, self._ng = n, nl, ng
        self._v_base = model.nodes[int(source.nodes[0])].kv_base_ln * 1e3
        y_base = S_BASE_VA / self._v_base**2
        e = source.e_v / self._v_base  # pu at 1 pu
        ys = source.y_s / y_base
        src = source.nodes
        k = len(src)
        ys_nodes = sp.csr_matrix((ys.ravel(), (np.repeat(src, k), np.tile(src, k))), shape=(n, n))
        y = (sp.csr_matrix(model.ybus()) / y_base + ys_nodes).tocsc()
        b1 = np.zeros(n, dtype=np.complex128)
        b1[src] = ys @ e
        self._e_node = _flat_start(model, source, e)
        pairs = self.pairs

        cl, cg = _incidence(n, _ends(loads)), _incidence(n, _ends(pvs))
        zero = sp.csc_matrix((n, 2 * ng))
        a_kcl = sp.bmat(
            [[y.real, -y.imag, cl, None, -cg, None, zero],
             [y.imag, y.real, None, cl, None, -cg, None]]
        )  # fmt: skip
        d = _incidence(n, pairs).T  # V_p - V_n per pair

        # Device rows, bounds and objective in SX; KCL as one sparse constant in MX: through SX
        # AD its Jacobian costs one colour per nonzero of the densest row (hub buses).
        nx = 2 * n + 2 * nl + 4 * ng
        x = ca.SX.sym("x", nx)
        par = ca.SX.sym("p", 1 + 2 * (1 + ng))
        vr, vi = x[:n], x[n : 2 * n]
        lr, li = x[2 * n : 2 * n + nl], x[2 * n + nl : 2 * n + 2 * nl]
        o = 2 * n + 2 * nl
        gr, gi, gp, gq = (x[o + j * ng : o + (j + 1) * ng] for j in range(4))
        ulr, uli = ca.mtimes(_dm(cl.T), vr), ca.mtimes(_dm(cl.T), vi)
        ugr, ugi = ca.mtimes(_dm(cg.T), vr), ca.mtimes(_dm(cg.T), vi)
        dvr, dvi = ca.mtimes(_dm(d), vr), ca.mtimes(_dm(d), vi)
        v_pu = par[0]
        er, ei = v_pu * ca.DM(e.real), v_pu * ca.DM(e.imag)
        dr, di = er - vr[src.tolist()], ei - vi[src.tolist()]
        ysr, ysi = ca.DM(ys.real), ca.DM(ys.imag)
        isr = ca.mtimes(ysr, dr) - ca.mtimes(ysi, di)
        isi = ca.mtimes(ysi, dr) + ca.mtimes(ysr, di)
        p_k = ca.vertcat(ca.dot(er, isr) + ca.dot(ei, isi), gp)
        c1, c2 = par[1 : 2 + ng], par[2 + ng :]
        f = ca.dot(c1, p_k) + ca.dot(c2, p_k**2)
        h = ca.vertcat(
            ulr * lr + uli * li, uli * lr - ulr * li,  # load P, Q
            ugr * gr + ugi * gi - gp, ugi * gr - ugr * gi - gq,  # PV p, q
            dvr**2 + dvi**2,  # |V_p - V_n|^2
            gp**2 + gq**2,  # inverter rating
        )  # fmt: skip
        xm, pm = ca.MX.sym("x", nx), ca.MX.sym("p", par.numel())
        f_m, h_m = ca.Function("fh", [x, par], [f, h])(xm, pm)
        g = ca.vertcat(ca.mtimes(_dm(a_kcl), xm) - pm[0] * ca.DM(np.r_[b1.real, b1.imag]), h_m)
        opts: dict[str, Any] = {"print_time": False, "error_on_fail": False}
        opts |= {"ipopt.print_level": 0, "ipopt.sb": "yes"}
        opts |= {f"ipopt.{key}": value for key, value in IPOPT_OPTIONS.items()}
        opts["ipopt.max_wall_time"] = time_limit_s
        self._solver = ca.nlpsol("opf", "ipopt", {"x": xm, "p": pm, "f": f_m, "g": g}, opts)
        self._vpn2 = ((self.vpn_lo_v / self._v_base) ** 2, (self.vpn_hi_v / self._v_base) ** 2)

    def solve(
        self,
        *,
        source_v_pu: float,
        load_p_w: ArrayLike,
        load_q_var: ArrayLike,
        pv_p_avail_w: ArrayLike,
        pv_s_rated_va: ArrayLike,
        costs: Costs,
        pv_q_max_var: ArrayLike | None = None,
    ) -> OPFResult:
        """Solve one scenario from a flat start: source voltages ``source_v_pu · E``, neutral 0.

        Loads in :attr:`load_names` order, PV units in :attr:`pv_names` order (W, var, VA);
        ``pv_q_max_var`` bounds ``|q|`` per unit (default: the rating alone). The status is
        returned, never raised.
        """
        n, nl, ng, s = self._n, self._nl, self._ng, S_BASE_VA
        pl, ql = _vector(load_p_w, nl, "load_p_w") / s, _vector(load_q_var, nl, "load_q_var") / s
        p_avail = _vector(pv_p_avail_w, ng, "pv_p_avail_w") / s
        s_rated = _vector(pv_s_rated_va, ng, "pv_s_rated_va") / s
        q_max = s_rated if pv_q_max_var is None else _vector(pv_q_max_var, ng, "pv_q_max_var") / s
        c1, c2 = _vector(costs.c1, 1 + ng, "c1"), _vector(costs.c2, 1 + ng, "c2") * s
        free = np.full(2 * n + 2 * nl + 2 * ng, np.inf)
        lo, hi = self._vpn2
        zeros = np.zeros(2 * n)
        bounds = {
            "lbx": np.r_[-free, np.zeros(ng), -q_max],
            "ubx": np.r_[free, p_avail, q_max],
            "lbg": np.r_[zeros, pl, ql, np.zeros(2 * ng), lo,
                         np.full(ng, -np.inf)],
            "ubg": np.r_[zeros, pl, ql, np.zeros(2 * ng), hi, s_rated**2],
        }  # fmt: skip
        v0 = source_v_pu * self._e_node
        x0 = np.r_[v0.real, v0.imag, np.zeros(2 * nl + 4 * ng)]
        start = time.perf_counter()
        sol = self._solver(x0=x0, p=np.r_[source_v_pu, c1, c2], **bounds)
        t_solve = time.perf_counter() - start
        stats = self._solver.stats()
        x = np.asarray(sol["x"], dtype=np.float64).ravel()
        o = 2 * n + 2 * nl + 2 * ng
        return OPFResult(
            status=str(stats["return_status"]),
            objective=float(sol["f"]) * s,
            iterations=int(stats["iter_count"]),
            t_solve_s=t_solve,
            v=(x[:n] + 1j * x[n : 2 * n]).astype(np.complex128) * self._v_base,
            pv_p_w=x[o : o + ng] * s,
            pv_q_var=x[o + ng :] * s,
        )


def _devices(model: ElectricalModel, element_class: str) -> list[Element]:
    found = [e for e in model.elements if e.enabled and e.element_class == element_class]
    return sorted(found, key=lambda e: e.name)


def _ends(devices: list[Element]) -> NDArray[np.int64]:
    """``(m, 2)``: the two nodes of each two-node device."""
    return np.array([e.local_nodes() for e in devices], dtype=np.int64).reshape(-1, 2)


def _incidence(n: int, ends: NDArray[np.int64]) -> sp.csc_matrix:
    """``n x m``: +1 at the first node of each pair, -1 at its second; nothing at ground."""
    cols = np.r_[np.arange(len(ends)), np.arange(len(ends))]
    rows, vals = ends.T.ravel(), np.repeat([1.0, -1.0], len(ends))
    keep = rows != GROUND
    return sp.csc_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, len(ends)))


def vpn_bounds(
    model: ElectricalModel, vmin_pu: float, vmax_pu: float
) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
    """:func:`phase_neutral_pairs` and the bounds (V) of their voltage magnitudes: ``vmin_pu``
    and ``vmax_pu`` times the phase node's nominal line-to-neutral voltage."""
    pairs = phase_neutral_pairs(model)
    nominal = np.array([model.nodes[int(i)].nominal_v_ln for i in pairs[:, 0]], dtype=np.float64)
    return pairs, vmin_pu * nominal, vmax_pu * nominal


def phase_neutral_pairs(model: ElectricalModel) -> NDArray[np.int64]:
    """``(k, 2)``: each phase node and its bus's neutral node (:data:`GROUND` if none)."""
    by_bus: dict[str, dict[str, int]] = {}
    for node in model.nodes:
        by_bus.setdefault(node.bus, {})[node.conductor] = node.idx
    pairs = [
        (idx, nodes.get("n", GROUND))
        for nodes in by_bus.values()
        for conductor, idx in nodes.items()
        if conductor != "n"
    ]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def _flat_start(
    model: ElectricalModel, source: Source, e: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Per node, the ideal source voltage of its conductor (pu at 1 pu); 0 on neutrals."""
    by_conductor = {
        model.nodes[int(i)].conductor: ek for i, ek in zip(source.nodes, e, strict=True)
    }
    return np.array([by_conductor.get(node.conductor, 0.0) for node in model.nodes], dtype=complex)


def _vector(values: ArrayLike, n: int, name: str) -> FloatArray:
    out = np.asarray(values, dtype=np.float64).ravel()
    if out.shape != (n,):
        raise ValueError(f"{name}: expected {n} values, got {out.size}")
    return out


def _dm(a: sp.spmatrix) -> ca.DM:
    """A SciPy sparse matrix as a CasADi sparse ``DM``."""
    a = sp.csc_matrix(a)
    a.eliminate_zeros()
    a.sort_indices()
    sparsity = ca.Sparsity(a.shape[0], a.shape[1], a.indptr.tolist(), a.indices.tolist())
    return ca.DM(sparsity, a.data.tolist())
