"""Positive-sequence view of a variant, in gridfm-datakit's format.

**When.** After generation: ``export-posseq`` builds the view from a dataset's stored truth
(``views/stored.py``), so that a dataset holds the three-phase model only.

**Backbone.** The buses reached from the source through series PD elements that together cover
phases a, b and c at both ends (a bank of single-phase regulators counts as one connection). Every
other bus belongs to the backbone bus from which a breadth-first search over all series PD
elements reaches it first (``mapping``).

**Sequence parameters**, from the full model's primitive admittances with the neutral conductors at
0 V (the assumption of the Kron view), with ``a1 = [1, a^2, a]``, ``a = exp(j 2 pi / 3)`` and the
projection ``P(M) = (1/3) a1^H M a1`` of a 3x3 block:

- lines, switches and series reactors between two backbone buses: ``z1 = P(Z)`` of the phase
  impedance ``Z = Y_pp^-1`` (``Y_pp``: phase block of the series admittance, i.e. the
  Kron-reduced impedance) and ``P`` of the shunt admittance;
- transformers (with any jumper in parallel): the positive-sequence Thevenin two-port of their
  phase-node admittance, ``V1_t = h11 V1_f + h12 I1_t`` with ``h11 = -P(Y_tt^-1 Y_tf)`` and
  ``h12 = P(Y_tt^-1)`` (for a line without shunt ``h12 = P(Z)``, the rule above). The vector-group
  shift, ``angle(1 / h11)`` rounded to 30 deg, is removed (as in transmission planning cases) and
  the truth is rotated by the cumulative shift of its zone; the off-nominal ratio is ``|1 / h11|``
  and the series impedance ``h12``; the magnetising current is dropped;
- shunt PD elements at a backbone bus (capacitors): ``P`` of their phase admittance.

**Realisation.** The phase-a circuit of the balanced three-phase network, as a single-phase OpenDSS
circuit: balanced injections excite only the positive sequence, so its solution is the view's.
Ideal source at the source bus at ``|V1|`` of the full model, angle 0; one constant-power load
per backbone bus with the bus's *equivalent demand* in the full model's solution: the power that
the backbone elements deliver into the bus, i.e. everything that hangs from it (loads, DER,
laterals with their losses and shunts, earthing).

**Export.** datakit's columns (``io/datakit_columns.py``), per unit on ``BASE_MVA`` and the bus
line-to-line base, computed from the realised elements' admittances read back from the engine, so
that datakit's power balance holds on the view's own solution. DER enter the net demand ``Pd`` (a
datakit PQ bus has no generation); their output is kept in ``x_`` columns.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.views.stored import StoredScenario, StoredVariant

BASE_MVA = 100.0
"""Power base of the per-unit columns: MATPOWER's and graphkit's default ``baseMVA``."""
ALPHA = np.exp(2j * np.pi / 3)
A1 = np.array([1.0, ALPHA**2, ALPHA])
"""Positive-sequence phase vector a-b-c (b lags a by 120 degrees)."""
PHASES = ("a", "b", "c")
SQRT3 = math.sqrt(3.0)
SOURCE_X_OHM = 1e-9
"""Reactance of the realised source: |V| at the source bus equals its setpoint within 1e-9 pu."""
ANGLE_LIMIT_DEG = 360.0
VECTOR_GROUP_STEP_DEG = 30.0
MAX_RESIDUAL_SHIFT_DEG = 5.0
"""Angle of a transformer group beyond its vector group (an open-delta bank with unequal taps:
about 1.7 deg per 0.1 tap difference); more is a phase shifter, not handled."""
READBACK_TOL = 1e-9
"""Relative tolerance of the read-back checks of the realised elements."""
BALANCE_TOL = 1e-6
"""Largest power mismatch of a bus of the view's solution, relative to the network's power base:
beyond it a load left its voltage window (its power then differs by >= 36 % of its setpoint)."""

ViewStatus = Literal["converged", "off_setpoint", "not_converged", "error"]


class PosSeqError(RuntimeError):
    """A network the positive-sequence view does not handle."""


def projection(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Positive-sequence block ``(1/3) a1^H M_ij a1`` of every 3x3 block of a phase matrix."""
    k = m.shape[0] // 3
    out = np.empty((k, k), dtype=np.complex128)
    for i in range(k):
        for j in range(k):
            out[i, j] = A1.conj() @ m[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] @ A1 / 3
    return out


# ------------------------------------------------------------------------------------- backbone


@dataclass(frozen=True, slots=True)
class Group:
    """The series PD elements between two backbone buses: one branch of the view."""

    from_bus: str
    """The end nearer the source (breadth-first order)."""
    to_bus: str
    elements: tuple[int, ...]
    kind: Literal["line", "transformer"]


@dataclass(frozen=True, slots=True)
class Backbone:
    source: str
    buses: tuple[str, ...]
    """Breadth-first order from the source: the index is the view's bus number."""
    owner: dict[str, str]
    """Every bus of the network -> its backbone bus."""
    groups: tuple[Group, ...]
    shunts: tuple[tuple[str, tuple[int, ...]], ...]
    """(backbone bus, shunt PD elements with phase nodes there)."""

    def index(self) -> dict[str, int]:
        return {b: j for j, b in enumerate(self.buses)}


def _terminal_buses(e: Element, model: ElectricalModel) -> list[str | None]:
    out: list[str | None] = []
    for terminal in e.terminals:
        buses = {model.nodes[k].bus for k in terminal if k != GROUND}
        if len(buses) > 1:
            raise PosSeqError(f"{e.name}: a terminal on several buses {sorted(buses)}")
        out.append(next(iter(buses)) if buses else None)
    return out


def _phases(terminal: list[int], model: ElectricalModel) -> set[str]:
    return {model.nodes[k].conductor for k in terminal if k != GROUND} & set(PHASES)


def find_backbone(model: ElectricalModel) -> Backbone:
    """The three-phase backbone of a model and the backbone bus of every bus."""
    pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
    cover: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    multi: list[Element] = []
    shunts: dict[str, list[int]] = defaultdict(list)
    for e in model.elements:
        if not (e.is_pd and e.enabled):
            continue
        buses = _terminal_buses(e, model)
        distinct = sorted({b for b in buses if b is not None})
        if len(distinct) == 1:
            if any(_phases(t, model) for t in e.terminals):
                shunts[distinct[0]].append(e.idx)
            continue
        if len(distinct) > 2 or len(buses) > 2:
            multi.append(e)
        for i, (bi, ti) in enumerate(zip(buses, e.terminals, strict=True)):
            for bj, tj in list(zip(buses, e.terminals, strict=True))[i + 1 :]:
                if bi is None or bj is None or bi == bj:
                    continue
                key = (min(bi, bj), max(bi, bj))
                if e.idx not in pairs[key]:
                    pairs[key].append(e.idx)
                cover[key][bi] |= _phases(ti, model)
                cover[key][bj] |= _phases(tj, model)
    three = {key for key, c in cover.items() if all(set(PHASES) <= c[b] for b in key)}
    adjacency: dict[str, list[str]] = defaultdict(list)
    adjacency3: dict[str, list[str]] = defaultdict(list)
    for u, v in pairs:
        adjacency[u].append(v)
        adjacency[v].append(u)
        if (u, v) in three:
            adjacency3[u].append(v)
            adjacency3[v].append(u)
    sources = {model.nodes[k].bus for k in model.source_nodes()}
    if len(sources) != 1:
        raise PosSeqError(f"{model.network_id}: expected one source bus, got {sorted(sources)}")
    source = sources.pop()
    order = _bfs([source], adjacency3)
    if len(order) < 2:
        raise PosSeqError(f"{model.network_id}: no three-phase backbone from {source}")
    index = {b: j for j, b in enumerate(order)}
    owner = {b: b for b in order}
    queue = deque(order)
    while queue:
        u = queue.popleft()
        for v in sorted(adjacency[u]):
            if v not in owner:
                owner[v] = owner[u]
                queue.append(v)
    by_idx = {e.idx: e for e in model.elements}
    for e in multi:
        if sum(b in index for b in _terminal_buses(e, model) if b is not None) >= 2:
            raise PosSeqError(f"{e.name}: an element with more than two buses on the backbone")
    groups = []
    for key in sorted(three, key=lambda k: (index.get(k[0], -1), index.get(k[1], -1))):
        if not (key[0] in index and key[1] in index):
            continue
        u, v = sorted(key, key=index.__getitem__)
        classes = {by_idx[e].element_class for e in pairs[key]}
        # a transformer group may hold jumpers (IEEE 37: phase b through an open-delta regulator)
        kind: Literal["line", "transformer"] = "transformer" if "transformer" in classes else "line"
        groups.append(Group(u, v, tuple(sorted(pairs[key])), kind))
    groups.sort(key=lambda g: (index[g.to_bus], index[g.from_bus]))
    return Backbone(
        source=source,
        buses=tuple(order),
        owner=owner,
        groups=tuple(groups),
        shunts=tuple((b, tuple(shunts[b])) for b in order if b in shunts),
    )


def _bfs(start: list[str], adjacency: dict[str, list[str]]) -> list[str]:
    order, seen = list(start), set(start)
    queue = deque(start)
    while queue:
        u = queue.popleft()
        for v in sorted(adjacency[u]):
            if v not in seen:
                seen.add(v)
                order.append(v)
                queue.append(v)
    return order


# ------------------------------------------------------------------------- sequence parameters


class ModelIndex:
    """Lookups into a model: its elements by index and the phase nodes of every bus."""

    def __init__(self, model: ElectricalModel) -> None:
        self.model = model
        self.elements = {e.idx: e for e in model.elements}
        self.phase_nodes: dict[str, dict[str, int]] = defaultdict(dict)
        for n in model.nodes:
            if n.conductor in PHASES:
                self.phase_nodes[n.bus][n.conductor] = n.idx

    def block(self, elements: Sequence[int], buses: Sequence[str]) -> NDArray[np.complex128]:
        """Phase-node admittance (S) of ``elements`` on ``buses`` x (a, b, c); other positions
        (neutral, ground, other buses) are at 0 V."""
        position = {
            k: 3 * i + PHASES.index(c)
            for i, bus in enumerate(buses)
            for c, k in self.phase_nodes[bus].items()
        }
        y = np.zeros((3 * len(buses), 3 * len(buses)), dtype=np.complex128)
        for idx in elements:
            e = self.elements[idx]
            if e.yprim is None:
                raise PosSeqError(f"{e.name}: no primitive admittance")
            keep = [(r, position[k]) for r, k in enumerate(e.local_nodes()) if k in position]
            rows = np.array([r for r, _ in keep], dtype=np.intp)
            cols = np.array([q for _, q in keep], dtype=np.intp)
            np.add.at(y, (cols[:, None], cols[None, :]), e.yprim[np.ix_(rows, rows)])
        return y


@dataclass(frozen=True, slots=True)
class LineParams:
    z_ohm: complex
    """Series impedance per phase (positive sequence)."""
    b_s: float
    """Total shunt susceptance per phase."""


@dataclass(frozen=True, slots=True)
class TransformerParams:
    t: float
    """Off-nominal ratio at the from side (MATPOWER ``tap``)."""
    z_own: complex
    """Series impedance in per unit of ``s_ref_va`` and the bus bases, at the to side."""
    s_ref_va: float
    """Per-phase power base of ``z_own``."""
    phi_deg: float
    """Vector-group shift removed (a multiple of 30 deg): the to-side voltage lags by it in the
    full model."""


def line_params(ix: ModelIndex, g: Group) -> LineParams:
    y = ix.block(g.elements, (g.from_bus, g.to_bus))
    ys = -y[:3, 3:]
    z1 = complex(projection(np.asarray(np.linalg.inv(ys), dtype=np.complex128))[0, 0])
    shunt = complex(projection(y[:3, :3] - ys)[0, 0] + projection(y[3:, 3:] - ys)[0, 0])
    if abs(shunt.real) > READBACK_TOL * max(abs(shunt), abs(1 / z1)):
        raise PosSeqError(f"{g.from_bus}-{g.to_bus}: line shunt conductance {shunt.real:g} S")
    return LineParams(z1, shunt.imag)


def transformer_params(ix: ModelIndex, g: Group, v_ln: dict[str, float]) -> TransformerParams:
    """Parameters of a transformer group at the network state of ``ix``.

    The positive-sequence two-port seen by balanced load currents at the to side with a balanced
    voltage at the from side (Thevenin form of ``I_t = Y_tf V_f + Y_tt V_t``):
    ``V1_t = h11 V1_f + h12 I1_t`` with ``h11 = -P(Y_tt^-1 Y_tf)`` and ``h12 = P(Y_tt^-1)``. In per
    unit, ``1 / h11 = t exp(j phi)``: ``t`` is the no-load ratio (exact for any bank: a wye bank
    with unequal taps has ``1 / mean(t_p)``), ``phi`` the shift, kept as its vector-group part
    (rounded to 30 deg); ``h12`` is the series impedance. Unlike the projection of the
    admittance, this is not dominated by a jumper in parallel (IEEE 37)."""
    y = ix.block(g.elements, (g.from_bus, g.to_bus))
    inv_tt = np.linalg.pinv(y[3:, 3:])  # a delta winding floats: pinv drops the zero sequence
    h12 = complex(A1.conj() @ inv_tt @ A1 / 3)
    h11 = complex(-(A1.conj() @ inv_tt @ y[3:, :3] @ A1) / 3)
    vf, vt = v_ln[g.from_bus], v_ln[g.to_bus]
    ratio = vt / (h11 * vf)
    angle = math.degrees(np.angle(ratio))
    phi = VECTOR_GROUP_STEP_DEG * round(angle / VECTOR_GROUP_STEP_DEG)
    if abs(angle - phi) > MAX_RESIDUAL_SHIFT_DEG:
        raise PosSeqError(f"{g.from_bus}-{g.to_bus}: phase shift {angle:.3f} deg (phase shifter)")
    rating = sum(
        ix.elements[e].rating_va or 0.0
        for e in g.elements
        if ix.elements[e].element_class == "transformer"
    )
    s_ref = max(rating / 3, 1e3)
    z_own = h12 * s_ref / vt**2
    if z_own.real < 0:
        raise PosSeqError(f"{g.from_bus}-{g.to_bus}: active series impedance {z_own} pu")
    return TransformerParams(float(abs(ratio)), z_own, s_ref, (phi + 180.0) % 360.0 - 180.0)


def shunt_admittance(ix: ModelIndex, bus: str, elements: Sequence[int]) -> complex:
    """Positive-sequence admittance (S per phase) of shunt PD elements at a bus."""
    return complex(projection(ix.block(elements, (bus,)))[0, 0])


# --------------------------------------------------------------------------------- realisation


@dataclass(frozen=True, slots=True)
class Export:
    """The per-unit model of the realised view at one network state (datakit's columns)."""

    r: NDArray[np.float64]
    x: NDArray[np.float64]
    b: NDArray[np.float64]
    tap: NDArray[np.float64]
    y: tuple[NDArray[np.complex128], ...]
    """Yff, Yft, Ytf, Ytt."""
    gs: NDArray[np.float64]
    bs: NDArray[np.float64]
    ybus: sp.csr_matrix
    """Network admittance (per unit), sorted indices: the row-major order of datakit's table."""


@dataclass(frozen=True, slots=True, eq=False)
class PosSeqOutcome:
    status: ViewStatus
    v: NDArray[np.complex128] | None = None
    """Bus voltages of the view (pu, angle relative to the source)."""
    demand_va: NDArray[np.complex128] | None = None
    """Equivalent demand per bus (VA, three-phase), DER included."""
    slack_va: complex | None = None
    """Power the source delivers (VA, three-phase)."""
    truth: NDArray[np.complex128] | None = None
    """V1 of the full model per bus (pu), rotated to the source and to the view's zones."""
    message: str | None = None


class PosSeqView:
    """The positive-sequence view of a variant, solved for its stored scenarios."""

    def __init__(self, source: StoredVariant, config: GenerationConfig) -> None:
        self.source = source
        self.variant_id = source.variant_id
        model = source.model
        self.backbone = bb = find_backbone(model)
        self._index = bb.index()
        self.v_ln = {n.bus: n.kv_base_ln * 1e3 for n in model.nodes}
        self.v_ll = np.array([self.v_ln[b] * SQRT3 for b in bb.buses])
        limits = config.networks.voltage_limits
        self.vm_limits = (limits.vmin_pu, limits.vmax_pu)
        self.s_base_va = source.s_base_va
        ix = ModelIndex(model)
        self._ix = ix
        self._lines = {j: line_params(ix, g) for j, g in enumerate(bb.groups) if g.kind == "line"}
        self._xf = {j: g for j, g in enumerate(bb.groups) if g.kind == "transformer"}
        initial = {j: transformer_params(ix, g, self.v_ln) for j, g in self._xf.items()}
        self.shift_deg = self._zone_shifts({j: p.phi_deg for j, p in initial.items()})
        self.rate_mva = np.array([self._rating(g) for g in bb.groups])
        self._realised = {e for g in bb.groups for e in g.elements}
        self._realised |= {e for _, es in bb.shunts for e in es}
        """Elements the view realises: its branches and shunts."""
        self._demand, self._source_customers, self._source_pv = self._demand_rows()
        self._loaded = self._attached()
        self._der = self._der_owner()
        phase = {(n.bus, n.conductor): n.idx for n in model.nodes}
        self._phase_idx = np.array([[phase[(b, c)] for c in PHASES] for b in bb.buses])
        self._v_ln = np.array([self.v_ln[b] for b in bb.buses])
        self.ctx = DssContext.new()
        self._realise(config, initial)
        self._vidx: NDArray[np.intp] | None = None
        self._state: str | None = None
        self._exports: dict[str, Export] = {}
        self._unit_states: set[str] = set()
        self._initial = initial

    # ---------------------------------------------------------------------------- set-up

    def _zone_shifts(self, phi: dict[int, float]) -> NDArray[np.float64]:
        """Cumulative vector-group shift of every bus (deg): its voltage lags the source's by it."""
        bb = self.backbone
        shift = np.full(len(bb.buses), np.nan)
        shift[0] = 0.0
        children: dict[str, list[tuple[int, Group]]] = defaultdict(list)
        for j, g in enumerate(bb.groups):
            children[g.from_bus].append((j, g))
        queue = deque([bb.source])
        while queue:
            u = queue.popleft()
            for j, g in children[u]:
                k = self._index[g.to_bus]
                if np.isnan(shift[k]):
                    shift[k] = shift[self._index[u]] + phi.get(j, 0.0)
                    queue.append(g.to_bus)
        for j, g in enumerate(bb.groups):
            gap = shift[self._index[g.to_bus]] - shift[self._index[g.from_bus]] - phi.get(j, 0.0)
            if abs((gap + 180.0) % 360.0 - 180.0) > 1e-6:
                raise PosSeqError(f"{g.from_bus}-{g.to_bus}: vector-group shifts in a loop")
        return shift

    def _rating(self, g: Group) -> float:
        """MVA rating of a branch: sum over its elements (0 when unknown: unlimited)."""
        total = 0.0
        for idx in g.elements:
            e = self._ix.elements[idx]
            if g.kind == "transformer":
                total += e.rating_va if e.element_class == "transformer" and e.rating_va else 0.0
            elif e.rating_a:
                phases = len(_phases(e.terminals[0], self.source.model))
                total += e.rating_a * self.v_ln[g.from_bus] * phases
        return total / 1e6

    def _demand_rows(self) -> tuple[sp.csr_matrix, list[int], list[int]]:
        """``M`` with ``demand = M @ (V_local conj(I))`` over the extraction rows, and the
        customers and PV units at the source bus.

        At every bus but the source the demand is minus the power into the realised elements
        (what they deliver into the bus: power balance of the bus). At the source it is the power
        into the other PD elements plus the setpoints of the customers and PV units there; any
        other power-conversion element at the source bus is refused.
        """
        source, bb = self.source, self.backbone
        model, plan = source.model, source.plan
        realised = self._realised
        bus_of = {
            (e.idx, t): b for e in model.elements for t, b in enumerate(_terminal_buses(e, model))
        }
        rows, cols, vals = [], [], []
        for r, (e, t) in enumerate(zip(plan.element_idx, plan.terminal, strict=True)):
            bus = bus_of[(int(e), int(t))]
            if bus not in self._index:
                continue
            if bus == bb.source and int(e) not in realised:
                rows.append(0)
                cols.append(r)
                vals.append(1.0)
            elif bus != bb.source and int(e) in realised:
                rows.append(self._index[bus])
                cols.append(r)
                vals.append(-1.0)
        m = sp.csr_matrix((vals, (rows, cols)), shape=(len(bb.buses), len(plan.element_idx)))
        by_name = {e.name: e for e in model.elements}
        source_nodes = {n.idx for n in model.nodes if n.bus == bb.source}

        def at_source(name: str) -> bool:
            return any(k in source_nodes for t in by_name[name].terminals for k in t)

        customers = [j for j, name in enumerate(source.customer_elements) if at_source(name)]
        pv = [j for j, name in enumerate(source.pv_generators) if at_source(name)]
        known = {source.customer_elements[j] for j in customers}
        known |= {source.pv_generators[j] for j in pv}
        for e in model.elements:
            if e.is_pd or not e.enabled or e.element_class == "vsource":
                continue
            if at_source(e.name) and e.name not in known:
                raise PosSeqError(f"{e.name}: a power-conversion element at the source bus")
        return m, customers, pv

    def _attached(self) -> NDArray[np.intp]:
        """Backbone buses where an element that the view does not realise is connected (on any
        conductor): only these have a demand; elsewhere it is zero by the bus's power balance."""
        model, realised = self.source.model, self._realised
        attached = set()
        for e in model.elements:
            if not e.enabled or e.idx in realised or e.element_class == "vsource":
                continue
            for terminal in e.terminals:
                for k in terminal:
                    if k != GROUND and model.nodes[k].bus in self._index:
                        attached.add(self._index[model.nodes[k].bus])
        return np.array(sorted(attached), dtype=np.intp)

    def _der_owner(self) -> NDArray[np.int64]:
        """Backbone bus of every DER unit (PV and storage: the rows of ``scenarios/der``)."""
        model = self.source.model
        by_name = {e.name: e for e in model.elements}
        out = []
        for name in self.source.pv_generators:
            bus = _terminal_buses(by_name[name], model)[0]
            assert bus is not None
            out.append(self._index[self.backbone.owner[bus]])
        return np.array(out, dtype=np.int64)

    def _realise(self, config: GenerationConfig, xf: dict[int, TransformerParams]) -> None:
        bb, ctx = self.backbone, self.ctx
        freq = self.source.frequency_hz
        loads = config.normalisation.loads
        src = bb.source
        script = [
            f"Set DefaultBaseFrequency={freq!r}",
            f"New Circuit.posseq bus1={src}.1 phases=1 basekv={self.v_ln[src] / 1e3!r} pu=1 "
            f"angle=0 R1=0 X1={SOURCE_X_OHM!r} R0=0 X0={SOURCE_X_OHM!r}",
        ]
        for j, g in enumerate(bb.groups):
            if g.kind == "line":
                p = self._lines[j]
                c_nf = p.b_s / (2 * math.pi * freq) * 1e9
                script.append(
                    f"New Line.pb{j} bus1={g.from_bus}.1 bus2={g.to_bus}.1 phases=1 units=none "
                    f"length=1 rmatrix=[{p.z_ohm.real!r}] xmatrix=[{p.z_ohm.imag!r}] "
                    f"cmatrix=[{c_nf!r}]"
                )
            else:
                script.append(
                    f"New Transformer.pb{j} phases=1 windings=2 "
                    f"buses=[{g.from_bus}.1.0 {g.to_bus}.1.0] conns=[wye wye] "
                    f"kVs=[{self.v_ln[g.from_bus] / 1e3!r} {self.v_ln[g.to_bus] / 1e3!r}] "
                    f"%imag=0 %noloadloss=0 ppm_antifloat=0 " + _xf_args(xf[j])
                )
        for i, (bus, _) in enumerate(bb.shunts):
            script.append(
                f"New Capacitor.ps{i} bus1={bus}.1 phases=1 kV={self.v_ln[bus] / 1e3!r} kvar=0"
            )
        for i in self._loaded:
            bus = bb.buses[i]
            script.append(
                f"New Load.pd{i} bus1={bus}.1 phases=1 kV={self.v_ln[bus] / 1e3!r} kW=0 kvar=0 "
                f"model=1 vminpu={loads.vminpu!r} vmaxpu={loads.vmaxpu!r}"
            )
        for cmd in script:
            ctx.command(cmd)
        ctx.set_options(tolerance=self.source.tolerance, control_mode="OFF")
        self._shunt_kvar(self._shunts(self._ix), force=True)

    def _shunts(self, ix: ModelIndex) -> list[complex]:
        return [shunt_admittance(ix, bus, es) for bus, es in self.backbone.shunts]

    def _shunt_kvar(self, y: list[complex], *, force: bool = False) -> None:
        for i, ((bus, _), yi) in enumerate(zip(self.backbone.shunts, y, strict=True)):
            if abs(yi.real) > READBACK_TOL * abs(yi) or yi.imag < 0:
                raise PosSeqError(f"{bus}: shunt {yi} S is not a capacitor")
            kvar = yi.imag * self.v_ln[bus] ** 2 / 1e3
            if force or kvar:
                self.ctx.command(f"Edit Capacitor.ps{i} kvar={kvar!r}")

    # ------------------------------------------------------------------------------- per state

    def _apply_state(self, state_id: str) -> None:
        """Transformer and capacitor parameters of the full model at ``state_id``."""
        if state_id == self._state:
            return
        self._state = None  # until every edit is applied: an error leaves the circuit mixed
        ix = ModelIndex(self.source.state_model(state_id))
        for j, g in self._xf.items():
            p = transformer_params(ix, g, self.v_ln)
            self.ctx.command(f"Edit Transformer.pb{j} " + _xf_args(p))
        self._shunt_kvar(self._shunts(ix), force=True)
        self._state = state_id

    def _export(self) -> Export:
        """Per-unit model of the realised circuit, read back from the engine (after a solve)."""
        bb, ctx = self.backbone, self.ctx
        n, m = len(bb.buses), len(bb.groups)
        s_phase = BASE_MVA * 1e6 / 3
        r, x, b, tap = np.empty(m), np.empty(m), np.zeros(m), np.ones(m)
        for j, g in enumerate(bb.groups):
            cls = "Line" if g.kind == "line" else "Transformer"
            y2 = _phase_yprim(ctx, f"{cls}.pb{j}")
            base = np.array([self.v_ln[g.from_bus], self.v_ln[g.to_bus]])
            yp = y2 * np.outer(base, base) / s_phase
            if abs(yp[0, 1] - yp[1, 0]) > READBACK_TOL * abs(yp[0, 1]):
                raise PosSeqError(f"{cls}.pb{j}: realised two-port is not reciprocal")
            if g.kind == "line":
                ys = -yp[0, 1]
                b[j] = (yp[0, 0] + yp[1, 1] + 2 * yp[0, 1]).imag
            else:
                t = -yp[1, 1] / yp[0, 1]
                rank = abs(yp[0, 0] * yp[1, 1] - yp[0, 1] ** 2) / abs(yp[0, 1]) ** 2
                if abs(t.imag) > READBACK_TOL * abs(t) or rank > READBACK_TOL:
                    raise PosSeqError(f"Transformer.pb{j}: realised two-port is not t:1 + series")
                ys, tap[j] = yp[1, 1], t.real
            z = 1 / ys
            r[j], x[j] = z.real, z.imag
        yff, yft, ytf, ytt = branch_admittances(r, x, b, tap, np.zeros(m))
        gs, bs = np.zeros(n), np.zeros(n)
        for i, (bus, _) in enumerate(bb.shunts):
            y = _phase_yprim(ctx, f"Capacitor.ps{i}")[0, 0] * self.v_ln[bus] ** 2 / s_phase
            k = self._index[bus]
            gs[k] += y.real
            bs[k] += y.imag
        f = np.array([self._index[g.from_bus] for g in bb.groups], dtype=np.int64)
        t_ = np.array([self._index[g.to_bus] for g in bb.groups], dtype=np.int64)
        ybus = sp.csr_matrix(
            (
                np.concatenate([yff, yft, ytf, ytt, gs + 1j * bs]),
                (np.concatenate([f, f, t_, t_, np.arange(n)]),
                 np.concatenate([f, t_, f, t_, np.arange(n)])),
            ),
            shape=(n, n),
        )  # fmt: skip
        ybus.sum_duplicates()
        ybus.eliminate_zeros()
        ybus.sort_indices()
        return Export(r, x, b, tap, (yff, yft, ytf, ytt), gs, bs, ybus)

    # ------------------------------------------------------------------------------ per scenario

    def demand(self, outcome: StoredScenario) -> NDArray[np.complex128]:
        """Equivalent demand of every backbone bus (VA, three-phase) in a converged scenario."""
        plan = self.source.plan
        s_rows = (plan.select @ outcome.v) * np.conj(outcome.i_pd)
        full = np.asarray(self._demand @ s_rows, dtype=np.complex128)
        d = np.zeros_like(full)
        d[self._loaded] = full[self._loaded]
        for j in self._source_customers:
            d[0] += outcome.load_p_w[j] + 1j * outcome.load_q_var[j]
        if outcome.pv_p_w is not None and outcome.pv_q_var is not None:
            for j in self._source_pv:
                d[0] -= outcome.pv_p_w[j] + 1j * outcome.pv_q_var[j]
        return d

    def truth(self, outcome: StoredScenario) -> NDArray[np.complex128]:
        """V1 of the full model at every backbone bus (pu of the LN base), rotated so that the
        source has angle 0 and every zone by its vector-group shift."""
        v1 = outcome.v[self._phase_idx] @ A1.conj() / 3 / self._v_ln
        return np.asarray(
            v1 * np.exp(-1j * np.angle(v1[0])) * np.exp(1j * np.radians(self.shift_deg)),
            dtype=np.complex128,
        )

    def solve(self, outcome: StoredScenario) -> PosSeqOutcome:
        """The view of a scenario whose full model converged."""
        try:
            truth = self.truth(outcome)
            demand = self.demand(outcome)
            self._apply_state(outcome.state_id)
            api = self.ctx.api
            for i in self._loaded:
                api.Loads.Name(f"pd{i}")
                api.Loads.kW(float(demand[i].real) / 3e3)
                api.Loads.kvar(float(demand[i].imag) / 3e3)
            api.Vsources.First()
            api.Vsources.PU(float(abs(truth[0])))
            # the setters leave the loads' admittances at an earlier scenario's kW, and the
            # iterations would depend on history; the command rebuilds them
            self.ctx.command("Set LoadMult=1")
            self.ctx.command("Set Mode=Snap")
            status = self.ctx.solve()
            if not status.converged:
                return PosSeqOutcome("not_converged", truth=truth, demand_va=demand)
            if outcome.state_id not in self._exports:
                self._exports[outcome.state_id] = self._export()
            export = self._exports[outcome.state_id]
            self._unit_states.add(outcome.state_id)
            v = self._voltages()
            # power into the network at each bus (VA): minus the demand, the source's at bus 0
            s = v * np.conj(export.ybus @ v) * BASE_MVA * 1e6
            mismatch = np.abs(s[1:] + demand[1:])
            if mismatch.size and mismatch.max() > BALANCE_TOL * self.s_base_va:
                k = int(mismatch.argmax()) + 1
                message = f"bus {self.backbone.buses[k]}: power mismatch {mismatch[k - 1]:.3g} VA"
                return PosSeqOutcome("off_setpoint", truth=truth, demand_va=demand, message=message)
            slack = complex(s[0] + demand[0])
            return PosSeqOutcome("converged", v, demand, slack, truth)
        except Exception as exc:  # recorded, never dropped
            return PosSeqOutcome("error", message=f"{type(exc).__name__}: {exc}")

    def _voltages(self) -> NDArray[np.complex128]:
        """Bus voltages of the solved view (pu); the source's (OpenDSS's own power of the source
        across its 1e-9 ohm loses ~1e-3 to cancellation: the slack comes from the network side)."""
        if self._vidx is None:
            self._vidx = self.ctx.ynode_positions(f"{b}.1" for b in self.backbone.buses)
        return np.asarray(self.ctx.ynode_voltages()[self._vidx] / self._v_ln)

    # ---------------------------------------------------------------------------------- tables

    def mapping_table(self) -> pa.Table:
        bb = self.backbone
        lateral: dict[str, list[str]] = defaultdict(list)
        for bus, own in sorted(bb.owner.items()):
            if bus != own:
                lateral[own].append(bus)
        return pa.table(
            {
                "variant_id": [self.variant_id] * len(bb.buses),
                "posseq_bus": np.arange(len(bb.buses), dtype=np.int64),
                "backbone_bus": list(bb.buses),
                "aggregated_buses": [json.dumps(lateral[b]) for b in bb.buses],
            },
            schema=schema.POSSEQ_MAPPING,
        )

    def generators_table(self) -> pa.Table:
        """``views/posseq/generators``: the source, the only generator (its power is the slack's
        ``Pg``, ``Qg`` at bus 0 of each scenario)."""
        s_mva = self.s_base_va / 1e6
        return pa.Table.from_pylist(
            [
                {
                    "variant_id": self.variant_id, "idx": 0, "bus": 0,
                    "min_p_mw": -s_mva, "max_p_mw": s_mva,
                    "min_q_mvar": -s_mva, "max_q_mvar": s_mva,
                    "cp0_eur": 0.0, "cp1_eur_per_mw": 0.0, "cp2_eur_per_mw2": 0.0,
                    "in_service": 1, "is_slack_gen": 1,
                }
            ],
            schema=schema.POSSEQ_GENERATORS,
        )  # fmt: skip

    def begin_unit(self) -> None:
        """Start a work unit: :meth:`state_tables` then covers every state the unit meets."""
        self._unit_states = set()

    def state_tables(self) -> dict[str, pa.Table]:
        """``views/posseq/{buses,branches,ybus}`` of the states met since :meth:`begin_unit`."""
        parts: dict[str, list[pa.Table]] = defaultdict(list)
        n, m = len(self.backbone.buses), len(self.backbone.groups)
        types = np.zeros((n, 3), dtype=np.int8)
        types[1:, 0], types[0, 2] = 1, 1
        f = np.array([self._index[g.from_bus] for g in self.backbone.groups], dtype=np.int64)
        t = np.array([self._index[g.to_bus] for g in self.backbone.groups], dtype=np.int64)
        phi = np.array([self._initial[j].phi_deg if j in self._initial else 0.0 for j in range(m)])
        for sid in sorted(self._unit_states):
            ex = self._exports[sid]
            yff, yft, ytf, ytt = ex.y
            parts["views/posseq/buses"].append(
                pa.table(
                    {
                        "state_id": [sid] * n,
                        "bus": np.arange(n, dtype=np.int64),
                        "PQ": types[:, 0],
                        "PV": types[:, 1],
                        "REF": types[:, 2],
                        "vn_kv": self.v_ll / 1e3,
                        "min_vm_pu": np.full(n, self.vm_limits[0]),
                        "max_vm_pu": np.full(n, self.vm_limits[1]),
                        "GS": ex.gs,
                        "BS": ex.bs,
                    },
                    schema=schema.POSSEQ_BUSES,
                )
            )
            parts["views/posseq/branches"].append(
                pa.table(
                    {
                        "state_id": [sid] * m,
                        "idx": np.arange(m, dtype=np.int64),
                        "from_bus": f,
                        "to_bus": t,
                        "r": ex.r,
                        "x": ex.x,
                        "b": ex.b,
                        "Yff_r": yff.real,
                        "Yff_i": yff.imag,
                        "Yft_r": yft.real,
                        "Yft_i": yft.imag,
                        "Ytf_r": ytf.real,
                        "Ytf_i": ytf.imag,
                        "Ytt_r": ytt.real,
                        "Ytt_i": ytt.imag,
                        "tap": ex.tap,
                        "shift": np.zeros(m),
                        "ang_min": np.full(m, -ANGLE_LIMIT_DEG),
                        "ang_max": np.full(m, ANGLE_LIMIT_DEG),
                        "rate_a": self.rate_mva,
                        "br_status": np.ones(m, dtype=np.int8),
                        "x_vector_shift_deg": phi,
                    },
                    schema=schema.POSSEQ_BRANCHES,
                )
            )
            y = ex.ybus.tocoo()
            parts["views/posseq/ybus"].append(
                pa.table(
                    {
                        "state_id": [sid] * y.nnz,
                        "index1": y.row.astype(np.int64),
                        "index2": y.col.astype(np.int64),
                        "G": y.data.real,
                        "B": y.data.imag,
                    },
                    schema=schema.POSSEQ_YBUS,
                )
            )
        return {name: pa.concat_tables(tables) for name, tables in parts.items()}

    def tables(self, pairs: Sequence[tuple[StoredScenario, PosSeqOutcome]]) -> dict[str, pa.Table]:
        """``views/posseq/{bus_results,scenarios}`` of a chunk: what changes per scenario for the
        converged views, and the status of every view attempted."""
        status = pa.table(
            {
                "scenario_id": np.array([o.scenario_id for o, _ in pairs], dtype=np.int64),
                "variant_id": [o.variant_id for o, _ in pairs],
                "view_status": [p.status for _, p in pairs],
                "view_message": [p.message for _, p in pairs],
            },
            schema=schema.POSSEQ_SCENARIOS,
        )
        parts = [self._bus_results(o, p) for o, p in pairs if p.status == "converged"]
        results = pa.concat_tables(parts) if parts else schema.POSSEQ_BUS_RESULTS.empty_table()
        return {"views/posseq/bus_results": results, "views/posseq/scenarios": status}

    def _bus_results(self, o: StoredScenario, p: PosSeqOutcome) -> pa.Table:
        assert p.v is not None
        assert p.truth is not None
        assert p.demand_va is not None
        assert p.slack_va is not None
        n = len(self.backbone.buses)
        der_p, der_q, der_avail = (np.zeros(n) for _ in range(3))
        if o.pv_p_w is not None and o.pv_q_var is not None and o.pv_p_avail_w is not None:
            np.add.at(der_p, self._der, o.pv_p_w / 1e6)
            np.add.at(der_q, self._der, o.pv_q_var / 1e6)
            np.add.at(der_avail, self._der, o.pv_p_avail_w / 1e6)
        pg, qg = np.zeros(n), np.zeros(n)
        pg[0], qg[0] = p.slack_va.real / 1e6, p.slack_va.imag / 1e6
        return pa.table(
            {
                "scenario_id": np.full(n, o.scenario_id, dtype=np.int64),
                "bus": np.arange(n, dtype=np.int64),
                "Pd": p.demand_va.real / 1e6,
                "Qd": p.demand_va.imag / 1e6,
                "Pg": pg,
                "Qg": qg,
                "Vm": np.abs(p.v),
                "Va": _wrap_deg(np.degrees(np.angle(p.v))),
                "x_truth_Vm": np.abs(p.truth),
                "x_truth_Va": _wrap_deg(np.degrees(np.angle(p.truth))),
                "x_der_p_mw": der_p,
                "x_der_q_mvar": der_q,
                "x_der_p_avail_mw": der_avail,
            },
            schema=schema.POSSEQ_BUS_RESULTS,
        )


def _wrap_deg(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Angles in [-180, 180) degrees, as datakit."""
    return np.asarray((a + 180.0) % 360.0 - 180.0, dtype=np.float64)


def _xf_args(p: TransformerParams) -> str:
    r = p.z_own.real * 100
    return (
        f"kVAs=[{p.s_ref_va / 1e3!r} {p.s_ref_va / 1e3!r}] taps=[{p.t!r} 1] "
        f"XHL={p.z_own.imag * 100!r} %Rs=[{r / 2!r} {r / 2!r}]"
    )


def _phase_yprim(ctx: DssContext, name: str) -> NDArray[np.complex128]:
    """Primitive admittance (S) of a realised element without its ground positions."""
    y = ctx.element_yprim(name)
    ctx.activate_element(name)
    keep = [i for i, k in enumerate(ctx.api.CktElement.NodeOrder()) if k != 0]
    return y[np.ix_(keep, keep)]


def branch_admittances(
    r: NDArray[np.float64],
    x: NDArray[np.float64],
    b: NDArray[np.float64],
    tap: NDArray[np.float64],
    shift_deg: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], ...]:
    """Yff, Yft, Ytf, Ytt of MATPOWER branches (per unit): datakit's
    ``utils/power_balance.compute_branch_admittances`` (shift in degrees here)."""
    ys = np.asarray(1.0 / (r + 1j * x), dtype=np.complex128)
    t = tap * np.exp(1j * np.radians(shift_deg))
    ysh = 1j * b / 2
    yff = np.asarray((ys + ysh) / tap**2, dtype=np.complex128)
    return yff, -ys / np.conj(t), -ys / t, np.asarray(ys + ysh, dtype=np.complex128)
