"""Sampled perturbations of a variant and the OpenDSS script that realises them.

Each quantity draws from its own stream of the variant's seed, in name order:
- line length: ``Edit Line.x length=<L·a>`` scales the series impedance and the shunt admittance
  by ``a`` exactly;
- neutral impedance: the series matrix ``Z`` becomes ``D Z D`` with ``D = 1`` except ``sqrt(b)`` at
  the neutral conductors: neutral self terms times ``b``, phase-neutral terms times ``sqrt(b)``,
  positive definite for every ``b > 0``. Written as ``rmatrix``, ``xmatrix``, ``cmatrix`` (per the
  line's unit) followed by ``units`` and ``length``: matrices without them detach the linecode and
  rescale its capacitance wrongly;
- grounding resistance: ``R`` of a grounding reactor, ``Rneut`` of a grounded wye winding;
- phase reallocation: ``round(fraction · n)`` of the ``n`` eligible single-phase loads, chosen
  uniformly, each moved to one of the other phases of its bus, neutral kept;
- the substation (``variants.upstream``, :mod:`~gridfm_datakit_distr.variants.upstream`): the
  transformer rating; the source's sequence impedances follow from it, and the other feeders
  are added as loads at the source bus;
- the load level ``L`` (``scenarios.profiles.synthetic.variant_scale``): a factor on the
  variant's nominal loads, applied by the synthetic provider and to the substation's lumped
  loads; no edit.

Dispatchable generators are not drawn: one at every :data:`DG_EVERY`-th customer
(:func:`dg_sites`), at the network's capacity (``variants/dg.py``), the same in every variant.

:func:`check_edits` compares the line Yprims of the realised variant with the perturbation.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field

import numpy as np

from gridfm_datakit_distr.config.schema import PV, PVOrientation, Storage, Variants
from gridfm_datakit_distr.config.value_spec import Fixed, ValueSpec
from gridfm_datakit_distr.model.electrical import ComplexMatrix, ElectricalModel
from gridfm_datakit_distr.scenarios.seeds import VariantStream, variant_rng
from gridfm_datakit_distr.variants.targets import CustomerTarget, LineTarget, VariantTargets
from gridfm_datakit_distr.variants.upstream import (
    Upstream,
    draw_rating,
    load_commands,
    source_edit,
    source_impedance,
)

_IDENTITY = Fixed(1.0)
_NO_PV = Fixed(0.0)
DG_EVERY = 4
"""A dispatchable generator at every 4th load (Geth, Claeys and Deconinck, PSCC 2022)."""
_PV_KWP_BOUNDS = (4.0, 10.0, 150.0)
_STORAGE_KW = (4.0, 8.0, 25.0, 100.0)
"""SMART-DS customer batteries: the rating (kW) in each band of PV kWp between the bounds."""


@dataclass(frozen=True, slots=True)
class Substation:
    """The drawn substation of a variant (``variants.upstream``)."""

    site: Upstream
    rating_kva: float
    z1_ohm: complex
    """Positive- (and negative-) sequence impedance of the source (ohm)."""
    z0_ohm: complex
    """Zero-sequence impedance of the source (ohm)."""

    @property
    def loads(self) -> dict[str, tuple[float, float]]:
        """Lumped load -> nominal (P (W), Q (var)) of the other feeders on its phase."""
        site = self.site
        return {e: (site.p_w[k], site.q_var[k]) for k, e in enumerate(site.loads)}

    def record(self) -> dict[str, object]:
        """The ``upstream`` entry of ``variants.perturbation``."""
        return {
            "rating_kva": self.rating_kva,
            "z1_ohm": [self.z1_ohm.real, self.z1_ohm.imag],
            "z0_ohm": [self.z0_ohm.real, self.z0_ohm.imag],
            "loads": {e: list(pq) for e, pq in self.loads.items()},
        }


@dataclass(frozen=True, slots=True)
class Perturbation:
    """The sampled parameters of one attempt at a variant."""

    attempt: int
    line_length: dict[str, float] = field(default_factory=dict)
    """Line -> length factor."""
    neutral_impedance: dict[str, float] = field(default_factory=dict)
    """Line -> neutral impedance factor."""
    grounding_ohm: dict[str, float] = field(default_factory=dict)
    """Grounding point -> resistance (ohm)."""
    phases: dict[str, tuple[str, str]] = field(default_factory=dict)
    """Load -> (old bus1, new bus1)."""
    pv_penetration: float = 0.0
    """Drawn share of customers with PV (the realised share is ``len(pv_kwp) / customers``)."""
    pv_kwp: dict[str, float] = field(default_factory=dict)
    """Customer load -> peak power of its PV unit (kW)."""
    pv_power_factor: dict[str, float] = field(default_factory=dict)
    """Customer load -> power factor of its inverter, when not 1."""
    pv_reactive_fraction: dict[str, float] = field(default_factory=dict)
    """Customer load -> largest |q| of its inverter in the OPF (fraction of the rating), when
    drawn (a fixed spec is not stored per unit)."""
    pv_reactive_threshold: dict[str, float] = field(default_factory=dict)
    """Customer load -> available power (fraction of the rating) below which its inverter gives
    no reactive power in the OPF, when drawn."""
    storage_share: float = 0.0
    """Drawn share of the PV customers with a storage unit."""
    storage_kw: dict[str, float] = field(default_factory=dict)
    """Customer load -> power rating of its storage unit (kW)."""
    dg_kw: dict[str, float] = field(default_factory=dict)
    """Customer load -> capacity ``P^max`` of its dispatchable generator (kW), not drawn: the
    network's."""
    upstream: Substation | None = None
    """The substation behind the source (``variants.upstream``); None: the source as loaded."""
    variant_scale: float = 1.0
    """Load level ``L``: factor on every nominal load of the variant (1: not drawn)."""
    pv_orientation: dict[str, tuple[int, int]] = field(default_factory=dict)
    """Customer load -> (tilt, azimuth) of the PVGIS series its PV unit reads (degrees), with
    ``scenarios.pv_profile.pvgis.orientation``."""

    def to_json(self) -> str:
        """The ``perturbation`` column of ``variants.parquet``: the drawn quantities only."""
        data: dict[str, object] = {"attempt": self.attempt}
        for key, values in (
            ("line_length_scale", self.line_length),
            ("neutral_impedance_scale", self.neutral_impedance),
            ("grounding_resistance_ohm", self.grounding_ohm),
            ("phase_reallocation", {k: list(v) for k, v in self.phases.items()}),
            ("pv_kwp", self.pv_kwp),
            ("pv_power_factor", self.pv_power_factor),
            ("pv_reactive_fraction", self.pv_reactive_fraction),
            ("pv_reactive_threshold", self.pv_reactive_threshold),
            ("pv_orientation", {k: list(v) for k, v in self.pv_orientation.items()}),
            ("storage_kw", self.storage_kw),
            ("dg_kw", self.dg_kw),
        ):
            if values:
                data[key] = values
        if self.pv_penetration:
            data["pv_penetration"] = self.pv_penetration
        if self.storage_share:
            data["storage_share"] = self.storage_share
        if self.upstream is not None:
            data["upstream"] = self.upstream.record()
        if self.variant_scale != 1.0:
            data["variant_scale"] = self.variant_scale
        return json.dumps(data, sort_keys=True)


def sample_perturbation(
    config: Variants,
    targets: VariantTargets,
    seed: int,
    net: int,
    k: int,
    attempt: int = 0,
    *,
    der: PV | None = None,
    storage: Storage | None = None,
    dg_kw: float | None = None,
    upstream: Upstream | None = None,
    load_level: ValueSpec | None = None,
    orientation: PVOrientation | None = None,
) -> Perturbation:
    """Draw the perturbation of variant ``k`` of the network of key ``net`` (attempt
    ``attempt``); with ``der``, the PV units of its customers too, with ``storage`` the
    storage units of the customers with PV, with ``dg_kw`` a dispatchable generator of that
    capacity (kW) at each of :func:`dg_sites`, with ``upstream`` the network's substation, and
    with ``load_level`` its load level ``L``, and with an enabled ``orientation`` the orientation
    class of each PV unit."""
    spec = config.admittance
    line_length: dict[str, float] = {}
    if spec.line_length_scale != _IDENTITY:
        rng = variant_rng(seed, net, k, VariantStream.LINE_LENGTH, attempt)
        line_length = {t.element: float(spec.line_length_scale.sample(rng)) for t in targets.lines}
    neutral: dict[str, float] = {}
    if spec.neutral_impedance_scale != _IDENTITY:
        rng = variant_rng(seed, net, k, VariantStream.NEUTRAL_IMPEDANCE, attempt)
        neutral = {
            t.element: float(spec.neutral_impedance_scale.sample(rng))
            for t in targets.lines
            if t.neutral
        }
    grounding: dict[str, float] = {}
    if spec.grounding_resistance_ohm is not None:
        rng = variant_rng(seed, net, k, VariantStream.GROUNDING_RESISTANCE, attempt)
        grounding = {
            g.key: float(spec.grounding_resistance_ohm.sample(rng)) for g in targets.groundings
        }
    phases: dict[str, tuple[str, str]] = {}
    realloc = config.phase_reallocation
    if realloc.enabled and targets.loads:
        rng = variant_rng(seed, net, k, VariantStream.PHASE_REALLOCATION, attempt)
        n = len(targets.loads)
        chosen = sorted(
            int(c) for c in rng.choice(n, size=int(realloc.fraction * n + 0.5), replace=False)
        )
        for c in chosen:
            t = targets.loads[c]
            new = t.candidates[int(rng.integers(len(t.candidates)))]
            phases[t.element] = (t.bus1(t.phase), t.bus1(new))
    rho, kwp, pf, fraction, threshold = _pv(der, targets, seed, net, k, attempt)
    share, storage_kw = _storage(storage, targets, kwp, seed, net, k, attempt)
    dg = {c.element: dg_kw for c in dg_sites(targets)} if dg_kw is not None else {}
    substation = None
    if upstream is not None:
        rng = variant_rng(seed, net, k, VariantStream.UPSTREAM_RATING, attempt)
        kva = draw_rating(upstream.customers, rng)
        substation = Substation(upstream, kva, *source_impedance(kva, upstream.kv_ll))
    level = 1.0
    if load_level is not None and load_level != _IDENTITY:
        level = float(
            load_level.sample(variant_rng(seed, net, k, VariantStream.LOAD_LEVEL, attempt))
        )
    return Perturbation(
        attempt, line_length, neutral, grounding, phases, rho, kwp, pf, fraction, threshold,
        share, storage_kw, dg, substation, level,
        pv_orientation=_orientation(orientation, targets, kwp, seed, net, k, attempt),
    )  # fmt: skip


def dg_sites(targets: VariantTargets) -> tuple[CustomerTarget, ...]:
    """The customers with a dispatchable generator: the 4th, 8th, ... in the order of
    :attr:`VariantTargets.customers` (every enabled load, in name order). Our reading: the
    source gives neither an order nor a start."""
    return targets.customers[DG_EVERY - 1 :: DG_EVERY]


def storage_kw_for_pv(kwp: float) -> float:
    """Power rating (kW) of a storage unit at a customer with ``kwp`` kWp of PV, by the
    customer-battery table of NREL's SMART-DS (bands < 4, 4-10, 10-150, > 150 kW; each lower
    bound included, our reading)."""
    return _STORAGE_KW[bisect.bisect_right(_PV_KWP_BOUNDS, kwp)]


def _storage(
    spec: Storage | None,
    targets: VariantTargets,
    pv: dict[str, float],
    seed: int,
    net: int,
    k: int,
    attempt: int,
) -> tuple[float, dict[str, float]]:
    """Share, then one uniform draw per customer whatever the share and the PV placement; the
    rating from the customer's PV (``kw`` null) or drawn, one per customer: a unit at one share
    is there, with the same rating, at every higher one."""
    if spec is None or spec.share == _NO_PV or not pv:
        return 0.0, {}
    rng = variant_rng(seed, net, k, VariantStream.STORAGE_PLACEMENT, attempt)
    share = float(spec.share.sample(rng))
    chosen = rng.random(len(targets.customers)) < share
    placed = [
        c.element for c, on in zip(targets.customers, chosen, strict=True) if on and c.element in pv
    ]
    if spec.kw is None:
        return share, {e: storage_kw_for_pv(pv[e]) for e in placed}
    rng = variant_rng(seed, net, k, VariantStream.STORAGE_SIZE, attempt)
    sizes = {c.element: float(spec.kw.sample(rng)) for c in targets.customers}
    return share, {e: sizes[e] for e in placed}


def _pv(
    der: PV | None, targets: VariantTargets, seed: int, net: int, k: int, attempt: int
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Penetration, then one uniform draw and one size per customer whatever the penetration:
    a customer with PV at one penetration has it, with the same size, at every higher one."""
    if der is None or der.penetration == _NO_PV or not targets.customers:
        return 0.0, {}, {}, {}, {}
    rng = variant_rng(seed, net, k, VariantStream.DER_PLACEMENT, attempt)
    rho = float(der.penetration.sample(rng))
    chosen = rng.random(len(targets.customers)) < rho
    rng = variant_rng(seed, net, k, VariantStream.DER_SIZE, attempt)
    sizes = [float(der.kwp.sample(rng)) for _ in targets.customers]
    kwp = {c.element: s for c, s, on in zip(targets.customers, sizes, chosen, strict=True) if on}
    pf: dict[str, float] = {}
    if der.power_factor != _IDENTITY:
        rng = variant_rng(seed, net, k, VariantStream.DER_POWER_FACTOR, attempt)
        factors = [float(der.power_factor.sample(rng)) for _ in targets.customers]
        pf = {
            c.element: f for c, f, on in zip(targets.customers, factors, chosen, strict=True) if on
        }
    drawn: list[dict[str, float]] = []
    for spec, stream in (
        (der.reactive_fraction, VariantStream.DER_REACTIVE_FRACTION),
        (der.reactive_threshold, VariantStream.DER_REACTIVE_THRESHOLD),
    ):
        values: dict[str, float] = {}
        if not isinstance(spec, Fixed):
            rng = variant_rng(seed, net, k, stream, attempt)
            draws = [float(spec.sample(rng)) for _ in targets.customers]
            values = {
                c.element: v
                for c, v, on in zip(targets.customers, draws, chosen, strict=True)
                if on
            }
        drawn.append(values)
    return rho, kwp, pf, drawn[0], drawn[1]


def _orientation(
    spec: PVOrientation | None,
    targets: VariantTargets,
    pv: dict[str, float],
    seed: int,
    net: int,
    k: int,
    attempt: int,
) -> dict[str, tuple[int, int]]:
    """One pair per customer whatever the placement, uniform among :meth:`PVOrientation.pairs`;
    kept for the customers with PV, so a unit keeps its class at every higher penetration."""
    if spec is None or not spec.enabled or not pv:
        return {}
    pairs = spec.pairs()
    rng = variant_rng(seed, net, k, VariantStream.PV_ORIENTATION, attempt)
    drawn = rng.integers(len(pairs), size=len(targets.customers))
    return {
        c.element: pairs[j]
        for c, j in zip(targets.customers, drawn, strict=True)
        if c.element in pv
    }


def edit_script(
    p: Perturbation, targets: VariantTargets, window: tuple[float, float] = (0.6, 1.5)
) -> list[str]:
    """OpenDSS commands realising ``p``: lines, grounding points, loads, PV units, storage units,
    then dispatchable generators (on their customer's nodes, after its phase move; output set per
    scenario; constant power within the load window ``window``, pu), each in name order; then the
    substation: the source's impedances and the other feeders' loads (power set per scenario)."""
    script = []
    for t in targets.lines:
        alpha = p.line_length.get(t.element, 1.0)
        beta = p.neutral_impedance.get(t.element)
        if beta is not None:
            d = _neutral_scaling(t, beta)
            script.append(
                f"Edit {t.element} rmatrix={_lower(t.r * d)} xmatrix={_lower(t.x * d)} "
                f"cmatrix={_lower(t.c)} units={t.units} length={t.length * alpha!r}"
            )
        elif t.element in p.line_length:
            script.append(f"Edit {t.element} length={t.length * alpha!r}")
    for g in targets.groundings:
        if g.key in p.grounding_ohm:
            script.append(f"{g.command}{p.grounding_ohm[g.key]!r}")
    for load in targets.loads:
        if load.element in p.phases:
            script.append(f"Edit {load.element} bus1={p.phases[load.element][1]}")
    for prefix, placed in (("pv", p.pv_kwp), ("bess", p.storage_kw), ("dg", p.dg_kw)):
        for c in targets.customers:
            if c.element in placed:
                bus1 = p.phases[c.element][1] if c.element in p.phases else c.bus1
                script.append(
                    f"New Generator.{prefix}_{c.name} bus1={bus1} phases={c.phases} "
                    f"kV={c.kv!r} conn={c.conn} kW=0 kvar=0 model=1 vminpu={window[0]!r} "
                    f"vmaxpu={window[1]!r}"
                )
    if p.upstream is not None:
        up = p.upstream
        script += source_edit(up.site.vsource, up.z1_ohm, up.z0_ohm) + load_commands(up.site)
    return script


def check_edits(p: Perturbation, targets: VariantTargets, model: ElectricalModel) -> float:
    """Largest deviation of a line Yprim of ``model`` from the one ``p`` predicts, relative to
    its largest entry (round-off: ~1e-15)."""
    yprims = {e.name: e.yprim for e in model.elements}
    worst = 0.0
    for t in targets.lines:
        actual = yprims.get(t.element)
        if actual is None:
            return math.inf
        alpha = p.line_length.get(t.element, 1.0)
        beta = p.neutral_impedance.get(t.element)
        z, shunt = _pi_blocks(t.yprim)
        if beta is not None:
            z = z * _neutral_scaling(t, beta)
        expected = _pi_yprim(alpha * z, alpha * shunt)
        worst = max(worst, float(np.abs(actual - expected).max() / np.abs(expected).max()))
    return worst


def _neutral_scaling(t: LineTarget, beta: float) -> np.ndarray:
    d = np.ones(t.r.shape[0])
    d[list(t.neutral)] = math.sqrt(beta)
    return np.outer(d, d)


def _lower(m: np.ndarray) -> str:
    rows = [" ".join(repr(float(m[i, j])) for j in range(i + 1)) for i in range(m.shape[0])]
    return "[" + " | ".join(rows) + "]"


def _pi_blocks(y: ComplexMatrix) -> tuple[ComplexMatrix, ComplexMatrix]:
    n = y.shape[0] // 2
    y_ft = y[:n, n:]
    return np.asarray(np.linalg.inv(-y_ft), dtype=np.complex128), y[:n, :n] + y_ft


def _pi_yprim(z: ComplexMatrix, shunt: ComplexMatrix) -> ComplexMatrix:
    ys = np.asarray(np.linalg.inv(z), dtype=np.complex128)
    return np.block([[ys + shunt, -ys], [-ys, ys + shunt]])
