"""The OPF problem of one network state: a nodal current-voltage NLP solved with Ipopt (CasADi).

Unknowns: the node voltages in rectangular coordinates, one current per branch of every load,
storage unit, PV unit and dispatchable generator (DG), and the active and reactive power of each
PV unit and DG (the controls, PV units first). A storage unit is fixed at its set-point: a load of
minus its power. Constraints:

- Kirchhoff's current law at every node, ``(Y_state + Y_s) V - Y_s E = C_g I_g - C_l I_l``,
  linear, entered as one constant sparse matrix: ``Y_state`` is :meth:`ElectricalModel.ybus` (lines,
  switches, transformers at their taps, capacitors, reactors), ``Y_s`` and ``E`` the source's
  admittance and ideal voltages (:class:`Source`);
- each branch ``i, j`` of a device (:func:`~gridfm_datakit_distr.model.ground.unit_branches`:
  wye, delta, one to three phases): ``(V_i - V_j) conj(I) = (P + jQ) / n_branches``, the load's
  constant power (a storage unit's: minus its set-point), the control's ``p + jq``;
- PV units: ``0 <= p <= p_avail``, ``p^2 + q^2 <= s_rated^2``, ``|q| <= q_max``
  (the unit's reactive limit);
- DG: ``0 <= p <= p_max`` and ``0 <= q <= p_max``, box bounds without a rating circle (Geth,
  Claeys and Deconinck, PSCC 2022);
- ``v_min <= |V_p - V_n| <= v_max`` at every phase node of every bus (``|V_p|`` where the bus
  has no neutral node), per unit of the node's nominal voltage;
- ``|S| <= rating`` into winding 1 of every transformer with a rating.

Objective: ``sum_k (c2_k p_k^2 + c1_k p_k)`` over the source, the PV units and the DG
(:class:`Costs`),
``p_k`` the active power element ``k`` delivers; the source's is that of its ideal voltages,
``Re sum E conj(Y_s (E - V_s))``. The model is built once per network state; the source voltage
and the costs are CasADi parameters, the loads, PV limits and ratings enter as the bounds of the
call, so each scenario is one Ipopt solve from a flat start. The NLP is scaled to per unit on each
node's base voltage and a power base (:func:`power_base`); inputs and results are SI.

Requires the ``opf`` extra (CasADi).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from gridfm_datakit_distr.extract.source import Source
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.ground import conventional_zones, unit_branches
from gridfm_datakit_distr.opf.support import (
    BESS_PREFIX,
    DG_PREFIX,
    unsupported_reason,
    vpn_bounds,
)

S_BASE_VA = 1e3
"""Default and smallest power base of the NLP's scaling (VA): :func:`power_base`."""
IPOPT_OPTIONS: dict[str, Any] = {"tol": 1e-9, "max_iter": 3000}
"""With ``max_wall_time`` from :class:`OPFProblem` (a limit hit ends as
``Maximum_WallTime_Exceeded``)."""
SOLVED = "Solve_Succeeded"
"""The only Ipopt status that counts as solved."""

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Costs:
    """Coefficients of ``c2 p^2 + c1 p`` per controllable element: the source, then the PV units
    in :attr:`OPFProblem.pv_names` order, then the DG in :attr:`OPFProblem.dg_names` order; ``p``
    in W, ``c1`` per W, ``c2`` per W^2."""

    c1: FloatArray
    c2: FloatArray


DG_C1 = 0.1
"""``c1`` of a DG in the preset ``economic``: a tenth of the source's, as Geth, Claeys and
Deconinck (PSCC 2022, DG case study: "ten times smaller than the generation cost of the reference
bus generator")."""


def economic(n_pv: int, n_dg: int = 0) -> Costs:
    """Preset ``economic``: source ``c1 = 1``, DG ``c1 = DG_C1``, everything else 0. Without DG
    the objective is the power the source delivers (W): the minimum import. Export counts as
    negative import: with DG the optimum can export."""
    c1 = np.r_[1.0, np.zeros(n_pv), np.full(n_dg, DG_C1)]
    return Costs(c1=c1, c2=np.zeros(1 + n_pv + n_dg))


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
    source_p_w: float
    """Active power the source delivers at the returned point (W): that of its ideal voltages,
    behind its impedance (the ``p`` of the source's cost)."""
    dg_p_w: FloatArray = field(default_factory=lambda: np.zeros(0))
    """Active power each DG injects (W), in :attr:`OPFProblem.dg_names` order."""
    dg_q_var: FloatArray = field(default_factory=lambda: np.zeros(0))
    """Reactive power each DG injects (var)."""

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
        s_base_va: float = S_BASE_VA,
    ) -> None:
        """``vmin_pu``, ``vmax_pu``: bounds of the phase-to-neutral voltage magnitudes, per unit
        of each phase node's nominal line-to-neutral voltage, and of the phase-to-phase ones in a
        zone whose voltage to ground is a numerical convention (:func:`vpn_bounds`);
        ``time_limit_s``: wall-clock limit of one solve (s); ``s_base_va``: power base of the NLP's
        scaling (:func:`power_base`), with no effect on the problem itself."""
        reason = unsupported_reason(model)
        if reason is not None:
            raise ValueError(f"{model.variant_id} is outside the OPF's scope: {reason}")
        self.pairs, self.vpn_lo_v, self.vpn_hi_v = vpn_bounds(model, vmin_pu, vmax_pu)
        """Phase and neutral node (or second phase) of every bounded voltage, and its bounds (V)."""
        self.conventional = [np.array(z, dtype=np.int64) for z in conventional_zones(model)]
        """Zones whose voltage to ground is a numerical convention: only differences count."""
        generators = _devices(model, "generator")
        bess = [e for e in generators if e.name.lower().startswith(BESS_PREFIX)]
        dgs = [e for e in generators if e.name.lower().startswith(DG_PREFIX)]
        pvs = [e for e in generators if not e.name.lower().startswith((BESS_PREFIX, DG_PREFIX))]
        loads = _devices(model, "load")
        self.load_names = tuple(e.name for e in loads)
        """Load elements by name: the order of the scenario's customers."""
        self.bess_names = tuple(e.name for e in bess)
        """Storage generators by name, fixed injections: the order of the variant's units."""
        self._n_customers = len(loads)
        loads = loads + bess  # a storage unit is a load of minus its set-point
        self.pv_names = tuple(e.name for e in pvs)
        """PV generators by name: the order of the variant's PV units."""
        self.dg_names = tuple(e.name for e in dgs)
        """Dispatchable generators by name, the controls after the PV units: the order of the
        network's DG."""
        self._n_pv = len(pvs)
        controls = pvs + dgs
        rated = [e for e in _devices(model, "transformer") if e.rating_va]
        self.transformer_names = tuple(e.name for e in rated)
        """Transformers whose apparent power at winding 1 is bounded by their rating."""
        n, nl, ng, nt = model.n_nodes, len(loads), len(controls), len(rated)
        lb, gb = _branches(model, loads), _branches(model, controls)
        self._lb, self._gb = lb, gb
        self._n, self._nl, self._ng = n, nl, ng
        self._load_share = _shares(lb, nl)  # (branches, loads): 1 / n_branches
        pv_share = _shares(gb, ng)
        self._s_base = s_base_va
        # Per unit on each node's own base voltage and s_base: Y_pu = D Y D / s_base, D the
        # diagonal of the bases; a device's current is per unit of its first node's base.
        vb = np.array([node.kv_base_ln * 1e3 for node in model.nodes], dtype=np.float64)
        self._vb = vb
        src = source.nodes
        e = source.e_v / vb[src]  # pu at 1 pu
        ys = vb[src][:, None] * source.y_s * vb[src][None, :] / s_base_va
        k = len(src)
        ys_nodes = sp.csr_matrix((ys.ravel(), (np.repeat(src, k), np.tile(src, k))), shape=(n, n))
        scale = sp.diags(vb)
        y = (scale @ sp.csr_matrix(model.ybus()) @ scale / s_base_va + ys_nodes).tocsc()
        b1 = np.zeros(n, dtype=np.complex128)
        b1[src] = ys @ e
        self._e_node = _flat_start(model, source, e)

        nlb, ngb = len(lb), len(gb)
        self._nlb, self._ngb = nlb, ngb
        cl, cg = _incidence(n, lb, vb), _incidence(n, gb, vb)
        zero = sp.csc_matrix((n, 2 * ng))
        a_kcl = sp.bmat(
            [[y.real, -y.imag, cl, None, -cg, None, zero],
             [y.imag, y.real, None, cl, None, -cg, None]]
        )  # fmt: skip
        d = _incidence(n, self.pairs, vb).T  # (V_p - V_n) / base_p per pair

        # Device rows, bounds and objective in SX; KCL as one sparse constant in MX: through SX
        # AD its Jacobian costs one colour per nonzero of the densest row (hub buses).
        nx = 2 * n + 2 * nlb + 2 * ngb + 2 * ng
        x = ca.SX.sym("x", nx)
        par = ca.SX.sym("p", 1 + 2 * (1 + ng))
        vr, vi = x[:n], x[n : 2 * n]
        o = 2 * n
        lr, li = x[o : o + nlb], x[o + nlb : o + 2 * nlb]
        o += 2 * nlb
        gr, gi = x[o : o + ngb], x[o + ngb : o + 2 * ngb]
        o += 2 * ngb
        gp, gq = x[o : o + ng], x[o + ng : o + 2 * ng]
        ulr, uli = ca.mtimes(_dm(cl.T), vr), ca.mtimes(_dm(cl.T), vi)
        ugr, ugi = ca.mtimes(_dm(cg.T), vr), ca.mtimes(_dm(cg.T), vi)
        dvr, dvi = ca.mtimes(_dm(d), vr), ca.mtimes(_dm(d), vi)
        share = _dm(pv_share)
        v_pu = par[0]
        er, ei = v_pu * ca.DM(e.real), v_pu * ca.DM(e.imag)
        dr, di = er - vr[src.tolist()], ei - vi[src.tolist()]
        ysr, ysi = ca.DM(ys.real), ca.DM(ys.imag)
        isr = ca.mtimes(ysr, dr) - ca.mtimes(ysi, di)
        isi = ca.mtimes(ysi, dr) + ca.mtimes(ysr, di)
        p_src = ca.dot(er, isr) + ca.dot(ei, isi)
        p_k = ca.vertcat(p_src, gp)
        c1, c2 = par[1 : 2 + ng], par[2 + ng :]
        f = ca.dot(c1, p_k) + ca.dot(c2, p_k**2)
        h = ca.vertcat(
            ulr * lr + uli * li, uli * lr - ulr * li,  # load branch P, Q
            ugr * gr + ugi * gi - ca.mtimes(share, gp),  # PV branch p
            ugi * gr - ugr * gi - ca.mtimes(share, gq),  # PV branch q
            dvr**2 + dvi**2,  # |V_p - V_n|^2
            gp**2 + gq**2,  # inverter rating
            *[_winding_s2(model, e, vb, s_base_va, vr, vi) for e in rated],  # |S_winding1|^2
        )  # fmt: skip
        self._s_rated2 = np.array([e.rating_va for e in rated], dtype=np.float64) ** 2
        self._nt = nt
        xm, pm = ca.MX.sym("x", nx), ca.MX.sym("p", par.numel())
        f_m, h_m = ca.Function("fh", [x, par], [f, h])(xm, pm)
        g = ca.vertcat(ca.mtimes(_dm(a_kcl), xm) - pm[0] * ca.DM(np.r_[b1.real, b1.imag]), h_m)
        opts: dict[str, Any] = {"print_time": False, "error_on_fail": False}
        opts |= {"ipopt.print_level": 0, "ipopt.sb": "yes"}
        opts |= {f"ipopt.{key}": value for key, value in IPOPT_OPTIONS.items()}
        opts["ipopt.max_wall_time"] = time_limit_s
        self._solver = ca.nlpsol("opf", "ipopt", {"x": xm, "p": pm, "f": f_m, "g": g}, opts)
        self._p_source = ca.Function("p_source", [x, par], [p_src])
        base_p = vb[self.pairs[:, 0]]
        self._vpn2 = ((self.vpn_lo_v / base_p) ** 2, (self.vpn_hi_v / base_p) ** 2)

    def _start_from(self, start: PowerFlowStart, load_pu: NDArray[np.complex128]) -> FloatArray:
        """The NLP's variables at a power-flow solution, per unit as the problem scales them."""
        v = start.v / self._vb
        il = _branch_currents(v, self._vb, self._lb, load_pu)
        p = np.r_[np.asarray(start.pv_p_w), np.asarray(start.dg_p_w)]
        g = (p + 1j * np.r_[np.asarray(start.pv_q_var), np.asarray(start.dg_q_var)]) / self._s_base
        ig = _branch_currents(v, self._vb, self._gb, g)
        parts = [v.real, v.imag, il.real, il.imag, ig.real, ig.imag, g.real, g.imag]
        return np.asarray(np.concatenate(parts), dtype=np.float64)

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
        start: PowerFlowStart | None = None,
        bess_p_w: ArrayLike = (),
        dg_p_max_w: ArrayLike = (),
    ) -> OPFResult:
        """Solve one scenario from a flat start (source voltages ``source_v_pu · E``, neutral 0),
        or from ``start``, a power-flow solution of the same scenario.

        Loads in :attr:`load_names` order, PV units in :attr:`pv_names` order (W, var, VA), each
        drawing an equal share through each of its branches; ``pv_q_max_var`` bounds ``|q|`` per
        unit (default: the rating alone); ``bess_p_w``: the storage units' fixed set-points in
        :attr:`bess_names` order (W, discharging positive, q = 0); ``dg_p_max_w``: the DG's
        capacities in :attr:`dg_names` order (W), the bound of both ``p`` and ``q``. The status is
        returned, never raised.
        """
        n, nc, ng, s = self._n, self._n_customers, self._ng, self._s_base
        nb, npv = self._nl - nc, self._n_pv
        nd = ng - npv
        pl = np.r_[_vector(load_p_w, nc, "load_p_w"), -_vector(bess_p_w, nb, "bess_p_w")] / s
        ql = np.r_[_vector(load_q_var, nc, "load_q_var"), np.zeros(nb)] / s
        p_max = _vector(dg_p_max_w, nd, "dg_p_max_w") / s
        p_avail = np.r_[_vector(pv_p_avail_w, npv, "pv_p_avail_w") / s, p_max]
        s_rated = _vector(pv_s_rated_va, npv, "pv_s_rated_va") / s
        q_pv = s_rated if pv_q_max_var is None else _vector(pv_q_max_var, npv, "pv_q_max_var") / s
        q_lo, q_hi = np.r_[-q_pv, np.zeros(nd)], np.r_[q_pv, p_max]
        s_rated = np.r_[s_rated, np.full(nd, np.inf)]  # DG: no rating circle
        c1, c2 = _vector(costs.c1, 1 + ng, "c1"), _vector(costs.c2, 1 + ng, "c2") * s
        free = np.full(2 * n + 2 * self._nlb + 2 * self._ngb, np.inf)
        plb, qlb = self._load_share @ pl, self._load_share @ ql
        lo, hi = self._vpn2
        zeros = np.zeros(2 * n)
        pv_rows = np.zeros(2 * self._ngb)
        t_hi = self._s_rated2 / s**2
        bounds = {
            "lbx": np.r_[-free, np.zeros(ng), q_lo],
            "ubx": np.r_[free, p_avail, q_hi],
            "lbg": np.r_[zeros, plb, qlb, pv_rows, lo, np.full(ng + self._nt, -np.inf)],
            "ubg": np.r_[zeros, plb, qlb, pv_rows, hi, s_rated**2, t_hi],
        }
        if start is None:
            v0 = source_v_pu * self._e_node
            x0 = np.r_[v0.real, v0.imag, np.zeros(2 * self._nlb + 2 * self._ngb + 2 * ng)]
        else:
            x0 = self._start_from(start, pl + 1j * ql)
        par = np.r_[source_v_pu, c1, c2]
        t0 = time.perf_counter()
        sol = self._solver(x0=x0, p=par, **bounds)
        t_solve = time.perf_counter() - t0
        stats = self._solver.stats()
        x = np.asarray(sol["x"], dtype=np.float64).ravel()
        o = 2 * n + 2 * self._nlb + 2 * self._ngb
        return OPFResult(
            status=str(stats["return_status"]),
            objective=float(sol["f"]) * s,
            iterations=int(stats["iter_count"]),
            t_solve_s=t_solve,
            v=(x[:n] + 1j * x[n : 2 * n]).astype(np.complex128) * self._vb,
            pv_p_w=x[o : o + npv] * s,
            pv_q_var=x[o + ng : o + ng + npv] * s,
            source_p_w=float(self._p_source(x, par)) * s,
            dg_p_w=x[o + npv : o + ng] * s,
            dg_q_var=x[o + ng + npv :] * s,
        )


@dataclass(frozen=True, slots=True, eq=False)
class PowerFlowStart:
    """A power-flow solution of the scenario as the NLP's start: node voltages (V), the PV
    units' and the DG's outputs (W, var), in the problem's orders."""

    v: NDArray[np.complex128]
    pv_p_w: FloatArray
    pv_q_var: FloatArray
    dg_p_w: FloatArray = field(default_factory=lambda: np.zeros(0))
    dg_q_var: FloatArray = field(default_factory=lambda: np.zeros(0))


def power_base(load_va: ArrayLike) -> float:
    """Power base (VA) of the NLP's scaling: the power of ten at or below a tenth of the total
    nominal load, at least 1 kVA. Measured: with the median load, MV networks of small
    loads (IEEE 34: 10 kVA, EDP industrial: 1 kVA) ended class-A scenarios in restoration
    failures. The total load itself (rounded to a power of ten) verifies as well on every network
    still supported (IEEE 13, 34, EULV, a CSIRO feeder, EDP industrial: 65 of 65 solved dispatches
    within 1e-6 pu); its failures were on IEEE 123, refused since for its floating bus. Neither was
    better on that survey (it moves one IEEE 13 scenario from the iteration limit to infeasible),
    so the tenth stays."""
    total = float(np.sum(np.abs(np.asarray(load_va, dtype=np.float64))))
    if total <= 0:
        return S_BASE_VA
    return max(S_BASE_VA, 10.0 ** np.floor(np.log10(total / 10.0)))


def _branch_currents(
    v: NDArray[np.complex128], vb: FloatArray, branches: NDArray[np.int64], power: ArrayLike
) -> NDArray[np.complex128]:
    """Per-unit branch currents at per-unit node voltages ``v`` for devices of per-unit powers
    ``power`` (an equal share per branch), as the NLP defines them."""
    s = np.asarray(power, dtype=np.complex128)
    owner = branches[:, 2]
    counts = np.bincount(owner, minlength=s.size)
    a, b = branches[:, 0], branches[:, 1]
    ground = b == GROUND
    vb_b = np.where(ground, 1.0, vb[np.maximum(b, 0)])
    dv = v[a] - np.where(ground, 0.0, vb_b / vb[a] * v[np.maximum(b, 0)])
    return np.asarray(np.conj(s[owner] / counts[owner] / dv), dtype=np.complex128)


def _devices(model: ElectricalModel, element_class: str) -> list[Element]:
    found = [e for e in model.elements if e.enabled and e.element_class == element_class]
    return sorted(found, key=lambda e: e.name)


def _branches(model: ElectricalModel, devices: list[Element]) -> NDArray[np.int64]:
    """``(m, 3)``: the two nodes of every branch of the devices and the device's position."""
    rows = []
    for j, e in enumerate(devices):
        branches = unit_branches(model, e)
        if branches is None:  # unsupported_reason has refused the network
            raise ValueError(f"{e.name}: connection not covered")
        rows += [(a, b, j) for a, b in branches]
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def _shares(branches: NDArray[np.int64], n_devices: int) -> sp.csr_matrix:
    """``(branches, devices)``: 1 / (number of branches of the device) where the branch is its."""
    owner = branches[:, 2]
    counts = np.bincount(owner, minlength=n_devices)
    return sp.csr_matrix(
        (1.0 / counts[owner], (np.arange(len(owner)), owner)), shape=(len(owner), n_devices)
    )


def _winding_s2(
    model: ElectricalModel,
    element: Element,
    vb: FloatArray,
    s_base: float,
    vr: ca.SX,
    vi: ca.SX,
) -> ca.SX:
    """``|S|^2`` (pu of ``s_base``) into winding 1 of a transformer: ``sum V_k conj((Y V)_k)``
    over the positions of its first terminal, ``Y`` its Yprim at the state, per unit on the
    nodes' bases ``vb``."""
    assert element.yprim is not None
    local = element.local_nodes()
    base = np.array([1.0 if node == GROUND else vb[node] for node in local])
    y = base[:, None] * element.yprim * base[None, :] / s_base
    zero = ca.SX(0)
    ur = ca.vertcat(*[zero if node == GROUND else vr[node] for node in local])
    ui = ca.vertcat(*[zero if node == GROUND else vi[node] for node in local])
    first = element.n_conductors
    g, b = ca.DM(y.real[:first]), ca.DM(y.imag[:first])
    ir, ii = ca.mtimes(g, ur) - ca.mtimes(b, ui), ca.mtimes(b, ur) + ca.mtimes(g, ui)
    s_re = ca.dot(ur[:first], ir) + ca.dot(ui[:first], ii)
    s_im = ca.dot(ui[:first], ir) - ca.dot(ur[:first], ii)
    return s_re**2 + s_im**2


def _incidence(n: int, ends: NDArray[np.int64], vb: FloatArray) -> sp.csc_matrix:
    """``n x m``: +1 at the first node ``a`` of each pair, ``-vb[b] / vb[a]`` at its second ``b``
    (-1 on one voltage level); nothing at ground. Its transpose gives ``(V_a - V_b) / vb[a]`` in
    per unit; its columns put a current per unit of ``vb[a]`` into the per-unit KCL rows."""
    ends = ends[:, :2]
    m = len(ends)
    a, b = ends[:, 0], ends[:, 1]
    keep = b != GROUND
    rows = np.r_[a, b[keep]]
    cols = np.r_[np.arange(m), np.arange(m)[keep]]
    vals = np.r_[np.ones(m), -vb[b[keep]] / vb[a[keep]]]
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, m))


def _flat_start(
    model: ElectricalModel, source: Source, e: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Per node, the ideal source voltage of its conductor (pu of the node's base, at 1 pu): the
    source's angle and 1 pu on every phase node, 0 on neutrals."""
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
