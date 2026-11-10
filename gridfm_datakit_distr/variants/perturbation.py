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
  uniformly, each moved to one of the other phases of its bus, neutral kept.

:func:`check_edits` compares the line Yprims of the realised variant with the perturbation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from gridfm_datakit_distr.config.schema import PV, Variants
from gridfm_datakit_distr.config.value_spec import Fixed
from gridfm_datakit_distr.model.electrical import ComplexMatrix, ElectricalModel
from gridfm_datakit_distr.scenarios.seeds import VariantStream, variant_rng
from gridfm_datakit_distr.variants.targets import LineTarget, VariantTargets

_IDENTITY = Fixed(1.0)
_NO_PV = Fixed(0.0)


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
        ):
            if values:
                data[key] = values
        if self.pv_penetration:
            data["pv_penetration"] = self.pv_penetration
        return json.dumps(data, sort_keys=True)


def sample_perturbation(
    config: Variants,
    targets: VariantTargets,
    seed: int,
    i: int,
    k: int,
    attempt: int = 0,
    *,
    der: PV | None = None,
) -> Perturbation:
    """Draw the perturbation of variant ``k`` of network ``i`` (attempt ``attempt``); with
    ``der``, the PV units of its customers too."""
    spec = config.admittance
    line_length: dict[str, float] = {}
    if spec.line_length_scale != _IDENTITY:
        rng = variant_rng(seed, i, k, VariantStream.LINE_LENGTH, attempt)
        line_length = {t.element: float(spec.line_length_scale.sample(rng)) for t in targets.lines}
    neutral: dict[str, float] = {}
    if spec.neutral_impedance_scale != _IDENTITY:
        rng = variant_rng(seed, i, k, VariantStream.NEUTRAL_IMPEDANCE, attempt)
        neutral = {
            t.element: float(spec.neutral_impedance_scale.sample(rng))
            for t in targets.lines
            if t.neutral
        }
    grounding: dict[str, float] = {}
    if spec.grounding_resistance_ohm is not None:
        rng = variant_rng(seed, i, k, VariantStream.GROUNDING_RESISTANCE, attempt)
        grounding = {
            g.key: float(spec.grounding_resistance_ohm.sample(rng)) for g in targets.groundings
        }
    phases: dict[str, tuple[str, str]] = {}
    realloc = config.phase_reallocation
    if realloc.enabled and targets.loads:
        rng = variant_rng(seed, i, k, VariantStream.PHASE_REALLOCATION, attempt)
        n = len(targets.loads)
        chosen = sorted(
            int(c) for c in rng.choice(n, size=int(realloc.fraction * n + 0.5), replace=False)
        )
        for c in chosen:
            t = targets.loads[c]
            new = t.candidates[int(rng.integers(len(t.candidates)))]
            phases[t.element] = (t.bus1(t.phase), t.bus1(new))
    rho, kwp, pf = _pv(der, targets, seed, i, k, attempt)
    return Perturbation(attempt, line_length, neutral, grounding, phases, rho, kwp, pf)


def _pv(
    der: PV | None, targets: VariantTargets, seed: int, i: int, k: int, attempt: int
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Penetration, then one uniform draw and one size per customer whatever the penetration:
    a customer with PV at one penetration has it, with the same size, at every higher one."""
    if der is None or der.penetration == _NO_PV or not targets.customers:
        return 0.0, {}, {}
    rng = variant_rng(seed, i, k, VariantStream.DER_PLACEMENT, attempt)
    rho = float(der.penetration.sample(rng))
    chosen = rng.random(len(targets.customers)) < rho
    rng = variant_rng(seed, i, k, VariantStream.DER_SIZE, attempt)
    sizes = [float(der.kwp.sample(rng)) for _ in targets.customers]
    kwp = {c.element: s for c, s, on in zip(targets.customers, sizes, chosen, strict=True) if on}
    pf: dict[str, float] = {}
    if der.power_factor != _IDENTITY:
        rng = variant_rng(seed, i, k, VariantStream.DER_POWER_FACTOR, attempt)
        factors = [float(der.power_factor.sample(rng)) for _ in targets.customers]
        pf = {
            c.element: f for c, f, on in zip(targets.customers, factors, chosen, strict=True) if on
        }
    return rho, kwp, pf


def edit_script(
    p: Perturbation, targets: VariantTargets, window: tuple[float, float] = (0.6, 1.5)
) -> list[str]:
    """OpenDSS commands realising ``p``: lines, grounding points, loads, PV units (on their
    customer's nodes, after its phase move; output set per scenario; constant power within the
    load window ``window``, pu), each in name order."""
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
    for c in targets.customers:
        if c.element in p.pv_kwp:
            bus1 = p.phases[c.element][1] if c.element in p.phases else c.bus1
            script.append(
                f"New Generator.pv_{c.name} bus1={bus1} phases={c.phases} kV={c.kv!r} "
                f"conn={c.conn} kW=0 kvar=0 model=1 vminpu={window[0]!r} vmaxpu={window[1]!r}"
            )
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
