"""PV units, storage units and dispatchable generators of a variant and their output per
scenario.

Placement is part of the variant (``variants/perturbation.py``): ``Generator.pv_<load>`` on its
customer's nodes, ``model=1`` (constant P and Q within the load window).

Output per scenario: ``p_avail = kWp · clear_sky(hour) · cloud_factor(cloudiness) · a``, injected
up to the inverter rating, ``p = min(p_avail, s_rated)`` with ``s_rated = inverter_kva_per_kwp ·
kWp`` (clipping; no curtailment in PF labels); ``q = -p · tan(acos(pf))`` (the inverter absorbs
reactive power below unit power factor), limited so that ``p² + q² ≤ s_rated²``. With
``der.pv.reactive_mode: uniform`` (power factor 1) ``q = u · q_lim`` instead, ``u`` uniform on
[-1, 1) per unit and scenario from ``PV_Q`` and ``q_lim = min(q_max_var(p_avail),
sqrt(s_rated² - p²))``, the
OPF's limit and the rating's circle: IEEE 1547-2018's constant reactive power mode with a setpoint
drawn per scenario (it does not follow the voltage). The hour is the
time of day of the load profile when the scenario has one (seconds since the profile start, taken as
midnight), else drawn uniformly from the scenario's ``PROFILE_TIME`` stream; cloudiness, one per
scenario for the whole network, from ``PV_OUTPUT``; the availability ``a``, one per unit and
scenario from ``PV_AVAILABILITY`` (a fixed value draws nothing).

``clear_sky`` is a synthetic equinox shape, ``max(0, cos(π (h - 12) / 12))``: sunrise 6:00, solar
noon 12:00, sunset 18:00, no location or date. ``cloud_factor`` is the Kasten-Czeplak relation for
global irradiance, ``1 - 0.75 · c^3.4`` with ``c`` the cloud cover fraction (Kasten and Czeplak,
Solar Energy 24, 1980).

With ``scenarios.pv_profile.provider: pvgis`` the output per kWp is one hour of the network's PVGIS
series instead (``scenarios/pvgis.py``): ``p_avail = kWp · P / 1 kW · a``, ``P`` the series' power
of 1 kWp in the record drawn from ``PV_TIME``, at the load profile's local time of day when the
scenario has one; no cloudiness is drawn. With ``pvgis.orientation`` each unit reads the series of
its orientation class (drawn per variant, ``variants/perturbation.py``) at that record, so the hour
and the weather stay shared. Clipping and reactive power as above.

Storage units (``Generator.bess_<load>``, placed with PV customers) are fixed at a set-point per
scenario, ``p = setpoint · kW``, ``setpoint`` drawn per unit from ``STORAGE_SETPOINT`` (positive:
discharging), ``q = 0``. ``p_avail_w`` of a storage unit is its set-point: it has nothing to
curtail.

With ``der.pv.trip`` a unit trips (``p = q = 0``) when a voltage its protection reads is beyond
the variant's over- or under-voltage threshold (:class:`VoltageTrip`; the loop is the runner's,
``solve/pf_batch.py``).

Dispatchable generators (``Generator.dg_<load>``, at every 4th customer, ``variants/dg.py``) are
set per scenario to ``p = setpoint · a · P^max``, ``setpoint`` drawn per unit from
``DG_SETPOINT``, ``q = 0``: the power flow's operating point; the OPF dispatches them within
``0 <= p, q <= a · P^max``. The availability ``a`` (0: out of service), one per unit and scenario
from ``DG_AVAILABILITY`` (a fixed value draws nothing); ``p_avail_w`` of a DG is ``a · P^max``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import PV, DGProfile, PVProfile, StorageProfile
from gridfm_datakit_distr.config.value_spec import Fixed, ValueSpec
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.ground import unit_branches
from gridfm_datakit_distr.model.units import SQRT3
from gridfm_datakit_distr.scenarios.pvgis import PVSeries, draw_record
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng
from gridfm_datakit_distr.variants.perturbation import Perturbation

DAY_S = 86400


def clear_sky(hour: float) -> float:
    """Synthetic clear-sky output (per unit of kWp) at a time of day (h)."""
    return max(0.0, math.cos(math.pi * (hour - 12.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0


def cloud_factor(cloudiness: float) -> float:
    """Share of clear-sky irradiance under a cloud cover fraction (Kasten and Czeplak)."""
    return 1.0 - 0.75 * cloudiness**3.4


@dataclass(frozen=True, slots=True)
class PVUnit:
    generator: str
    """``Generator.pv_<load>``."""
    load: str
    """The customer's ``Load.<name>``."""
    kwp: float
    s_rated_va: float
    power_factor: float
    reactive_fraction: float
    """Largest |q| in the OPF (and of a drawn q), as a fraction of ``s_rated_va``."""
    reactive_threshold: float
    """No reactive power in the OPF (nor a drawn q) below this available power (fraction of
    ``s_rated_va``)."""
    orientation: tuple[int, int] | None = None
    """(tilt, azimuth) of the PVGIS series the unit reads (degrees), with
    ``scenarios.pv_profile.pvgis.orientation``."""

    def q_max_var(self, p_avail_w: float) -> float:
        """The largest |q| (var) the OPF may dispatch at an available power (W); with
        ``reactive_mode: uniform`` also the bound of the power flow's drawn q."""
        if p_avail_w < self.reactive_threshold * self.s_rated_va:
            return 0.0
        return self.reactive_fraction * self.s_rated_va


def _fixed(spec: ValueSpec) -> float:
    """The value of a fixed spec (a drawn one is stored per unit in the perturbation)."""
    if not isinstance(spec, Fixed):
        raise ValueError(f"no drawn value for a unit, spec {spec!r}")
    return spec.value


def _per_unit(drawn: dict[str, float], load: str, spec: ValueSpec) -> float:
    """The unit's drawn value, else the value of a fixed spec (looked up first: a drawn spec has
    no fixed value)."""
    return drawn[load] if load in drawn else _fixed(spec)


def pv_units(p: Perturbation, spec: PV) -> tuple[PVUnit, ...]:
    """The PV units a variant's perturbation placed, ordered by customer."""
    return tuple(
        PVUnit(
            generator=f"Generator.pv_{load.split('.', 1)[1]}",
            load=load,
            kwp=kwp,
            s_rated_va=spec.inverter_kva_per_kwp * kwp * 1e3,
            power_factor=p.pv_power_factor.get(load, 1.0),
            reactive_fraction=_per_unit(p.pv_reactive_fraction, load, spec.reactive_fraction),
            reactive_threshold=_per_unit(p.pv_reactive_threshold, load, spec.reactive_threshold),
            orientation=p.pv_orientation.get(load),
        )
        for load, kwp in sorted(p.pv_kwp.items())
    )


@dataclass(frozen=True, slots=True, eq=False)
class PVDraw:
    """One scenario's PV setpoints, ordered as the units."""

    hour: float | None
    """Time of day of the clear-sky shape (h); None with a PVGIS series."""
    cloudiness: float | None
    """None with a PVGIS series."""
    p_avail_w: NDArray[np.float64]
    p_w: NDArray[np.float64]
    """Injected active power (generation positive)."""
    q_var: NDArray[np.float64]
    """Injected reactive power (negative: absorbed)."""
    time_utc_s: int | None = None
    """Start of the PVGIS hour drawn (Unix time, s); None with the clear-sky shape."""


class PVOutputs:
    """The outputs of a variant's PV units, scenario by scenario; ``series``: the network's
    PVGIS series (``provider: pvgis``), with ``orientation`` one per class (each unit reads its
    own at the scenario's hour), None for the synthetic clear sky."""

    def __init__(
        self,
        units: Sequence[PVUnit],
        spec: PV,
        profile: PVProfile,
        series: PVSeries | Mapping[tuple[int, int], PVSeries] | None = None,
    ) -> None:
        if (series is None) != (profile.provider != "pvgis"):
            raise ValueError(f"provider {profile.provider!r} with series {series!r}")
        self.units = tuple(units)
        self._unit_series: list[PVSeries] | None = None
        if isinstance(series, Mapping):
            classes = [u.orientation for u in self.units]
            if None in classes:
                raise ValueError("a series per orientation class needs units with a class")
            self._unit_series = [series[c] for c in classes if c is not None]
            series = next(iter(series.values()))  # the clock: one location, one set of years
            if any(
                u.start_s != series.start_s or len(u.p_w) != len(series.p_w)
                for u in self._unit_series
            ):
                raise ValueError("the orientation class series do not share one clock")
        elif any(u.orientation is not None for u in self.units):
            raise ValueError("units with an orientation class need a series per class")
        self.spec = spec
        self.profile = profile
        self.series = series
        self._kwp = np.array([u.kwp for u in self.units], dtype=np.float64)
        self._rated = np.array([u.s_rated_va for u in self.units], dtype=np.float64)
        self._tan = np.tan(np.arccos([u.power_factor for u in self.units]))

    def draw(self, seed: int, net: int, k: int, s: int, time_s: int | None) -> PVDraw:
        """Setpoints of scenario ``s``; ``time_s``: the load profile's time, if it has one."""
        if self.series is not None:
            return self._draw_series(self.series, seed, net, k, s, time_s)
        if time_s is None:
            hour = float(scenario_rng(seed, net, k, s, ScenarioStream.PROFILE_TIME).uniform(0, 24))
        else:
            hour = (int(time_s) % DAY_S) / 3600.0
        rng = scenario_rng(seed, net, k, s, ScenarioStream.PV_OUTPUT)
        cloudiness = float(self.profile.cloudiness.sample(rng))
        availability = self._availability(seed, net, k, s)
        p_avail = 1e3 * self._kwp * clear_sky(hour) * cloud_factor(cloudiness) * availability
        return PVDraw(hour, cloudiness, p_avail, *self._setpoints(p_avail, seed, net, k, s))

    def _draw_series(
        self, series: PVSeries, seed: int, net: int, k: int, s: int, time_s: int | None
    ) -> PVDraw:
        rng = scenario_rng(seed, net, k, s, ScenarioStream.PV_TIME)
        i = draw_record(series, rng, None if time_s is None else int(time_s) % DAY_S)
        per_kwp = (
            series.p_w[i]
            if self._unit_series is None
            else np.array([u.p_w[i] for u in self._unit_series])
        )
        p_avail = self._kwp * per_kwp * self._availability(seed, net, k, s)
        time_utc_s = series.start_s + 3600 * i
        p, q = self._setpoints(p_avail, seed, net, k, s)
        return PVDraw(None, None, p_avail, p, q, time_utc_s=time_utc_s)

    def _availability(self, seed: int, net: int, k: int, s: int) -> float | NDArray[np.float64]:
        spec = self.profile.availability
        if isinstance(spec, Fixed):
            return spec.value
        rng = scenario_rng(seed, net, k, s, ScenarioStream.PV_AVAILABILITY)
        return np.array([float(spec.sample(rng)) for _ in self.units])

    def _setpoints(
        self, p_avail: NDArray[np.float64], seed: int, net: int, k: int, s: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        p = np.minimum(p_avail, self._rated)  # inverter clipping when kVA < kWp
        q_limit = np.sqrt(np.maximum(self._rated**2 - p**2, 0.0))
        if self.spec.reactive_mode == "uniform":
            q_opf = [u.q_max_var(float(a)) for u, a in zip(self.units, p_avail, strict=True)]
            rng = scenario_rng(seed, net, k, s, ScenarioStream.PV_Q)
            return p, rng.uniform(-1.0, 1.0, len(self.units)) * np.minimum(q_opf, q_limit)
        q = -np.minimum(p * self._tan, q_limit)
        return p, q


def protection_pairs(model: ElectricalModel, element: Element) -> list[tuple[int, int]]:
    """The node pairs whose voltages a unit's protection reads (:data:`GROUND` allowed second):
    its own terminal branches (``unit_branches``: wye, each phase to its neutral node or to
    ground; delta, phase to phase) and, for a wye unit on several phases, phase to phase too
    (EN 50549-1 4.9.3). ``ValueError`` for a connection ``unit_branches`` does not cover."""
    branches = unit_branches(model, element)
    if branches is None:
        raise ValueError(f"{element.name}: connection not covered by the voltage trip")
    if len(branches) >= 2 and all(
        q == GROUND or model.nodes[q].conductor == "n" for _, q in branches
    ):
        phases = [p for p, _ in branches]
        ring = list(zip(phases, phases[1:] + phases[:1], strict=True))
        branches += ring if len(phases) > 2 else ring[:1]
    return branches


def pair_base_v(model: ElectricalModel, p: int, q: int) -> float:
    """Nominal voltage (V) of a pair: the phase node's nominal line-to-neutral voltage to a
    neutral or ground; between two phases √3 times it, across the two halves of a split-phase
    service (s1, s2) 2 times it."""
    ln = model.nodes[p].nominal_v_ln
    if q == GROUND or model.nodes[q].conductor == "n":
        return ln
    if {model.nodes[p].conductor, model.nodes[q].conductor} == {"s1", "s2"}:
        return 2.0 * ln
    return SQRT3 * ln


class VoltageTrip:
    """Over- and under-voltage disconnection of a variant's PV units: a unit trips when any
    voltage of :func:`protection_pairs`, per unit of :func:`pair_base_v`, is above ``ov_pu`` or
    below ``uv_pu`` (snapshot thresholds: no delay)."""

    def __init__(
        self, model: ElectricalModel, units: Sequence[PVUnit], ov_pu: float, uv_pu: float
    ) -> None:
        elements = {e.name.lower(): e for e in model.elements}
        n, rows = len(model.nodes), []
        for i, u in enumerate(units):
            for p, q in protection_pairs(model, elements[u.generator.lower()]):
                rows.append((p, n if q == GROUND else q, i, pair_base_v(model, p, q)))
        self.ov_pu, self.uv_pu, self.n_units = ov_pu, uv_pu, len(units)
        self._p = np.array([r[0] for r in rows], dtype=np.int64)
        self._q = np.array([r[1] for r in rows], dtype=np.int64)
        """Second node of each pair; ``n`` (one past the nodes) is ground, at 0 V."""
        self._unit = np.array([r[2] for r in rows], dtype=np.int64)
        self._base = np.array([r[3] for r in rows], dtype=np.float64)

    def beyond(self, v: NDArray[np.complex128]) -> NDArray[np.bool_]:
        """Per unit: whether a voltage it reads is beyond a threshold; ``v`` the node voltages
        (V) in the model's node order."""
        vg = np.append(v, 0.0)
        pu = np.abs(vg[self._p] - vg[self._q]) / self._base
        out = (pu > self.ov_pu) | (pu < self.uv_pu)
        return np.bincount(self._unit[out], minlength=self.n_units) > 0


@dataclass(frozen=True, slots=True)
class RatedUnit:
    """A generator with a power rating at a customer: a storage unit or a DG."""

    generator: str
    """``Generator.bess_<load>`` or ``Generator.dg_<load>``."""
    load: str
    kw: float
    """Power rating (kW): a storage unit's rating, a DG's capacity ``P^max``."""


def _units(prefix: str, placed: dict[str, float]) -> tuple[RatedUnit, ...]:
    return tuple(
        RatedUnit(f"Generator.{prefix}_{load.split('.', 1)[1]}", load, kw)
        for load, kw in sorted(placed.items())
    )


def storage_units(p: Perturbation) -> tuple[RatedUnit, ...]:
    """The storage units a variant's perturbation placed, ordered by customer."""
    return _units("bess", p.storage_kw)


def dg_units(p: Perturbation) -> tuple[RatedUnit, ...]:
    """The dispatchable generators of a variant, ordered by customer."""
    return _units("dg", p.dg_kw)


class StorageSetpoints:
    """The set-points of a variant's storage units, scenario by scenario."""

    stream = ScenarioStream.STORAGE_SETPOINT

    def __init__(self, units: Sequence[RatedUnit], profile: StorageProfile | DGProfile) -> None:
        self.units = tuple(units)
        self.profile = profile
        self.rated_w = np.array([1e3 * u.kw for u in self.units], dtype=np.float64)

    def capacity(self, seed: int, net: int, k: int, s: int) -> NDArray[np.float64]:
        """Each unit's capacity in scenario ``s`` (W): its rating."""
        return self.rated_w

    def draw(self, seed: int, net: int, k: int, s: int) -> NDArray[np.float64]:
        """Active power of each unit in scenario ``s`` (W, discharging positive): the set-point
        times :meth:`capacity`."""
        spec = self.profile.setpoint
        capacity = self.capacity(seed, net, k, s)
        if isinstance(spec, Fixed):
            return spec.value * capacity
        rng = scenario_rng(seed, net, k, s, self.stream)
        return np.array([float(spec.sample(rng)) for _ in self.units]) * capacity


class DGSetpoints(StorageSetpoints):
    """The power-flow set-points of a variant's DG, scenario by scenario (W): a fraction of the
    available capacity drawn per unit from their own stream."""

    stream = ScenarioStream.DG_SETPOINT
    profile: DGProfile

    def capacity(self, seed: int, net: int, k: int, s: int) -> NDArray[np.float64]:
        """Each unit's available capacity ``a · P^max`` in scenario ``s`` (W), ``a`` drawn per
        unit from ``DG_AVAILABILITY`` (a fixed value draws nothing)."""
        spec = self.profile.availability
        if isinstance(spec, Fixed):
            return spec.value * self.rated_w
        rng = scenario_rng(seed, net, k, s, ScenarioStream.DG_AVAILABILITY)
        return np.array([float(spec.sample(rng)) for _ in self.units]) * self.rated_w
