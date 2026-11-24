"""PV units, storage units and dispatchable generators of a variant and their output per
scenario.

Placement is part of the variant (``variants/perturbation.py``): ``Generator.pv_<load>`` on its
customer's nodes, ``model=1`` (constant P and Q within the load window).

Output per scenario: ``p_avail = kWp · clear_sky(hour) · cloud_factor(cloudiness) · a``, injected
up to the inverter rating, ``p = min(p_avail, s_rated)`` with ``s_rated = inverter_kva_per_kwp ·
kWp`` (clipping; no curtailment in PF labels); ``q = -p · tan(acos(pf))`` (the inverter absorbs
reactive power below unit power factor), limited so that ``p² + q² ≤ s_rated²``. The hour is the
time of day of the load profile when the scenario has one (seconds since the profile start, taken as
midnight), else drawn uniformly from the scenario's ``PROFILE_TIME`` stream; cloudiness, one per
scenario for the whole network, from ``PV_OUTPUT``; the availability ``a``, one per unit and
scenario from ``PV_AVAILABILITY`` (a fixed value draws nothing).

``clear_sky`` is a synthetic equinox shape, ``max(0, cos(π (h - 12) / 12))``: sunrise 6:00, solar
noon 12:00, sunset 18:00, no location or date. ``cloud_factor`` is the Kasten-Czeplak relation for
global irradiance, ``1 - 0.75 · c^3.4`` with ``c`` the cloud cover fraction (Kasten and Czeplak,
Solar Energy 24, 1980).

Storage units (``Generator.bess_<load>``, placed with PV customers) are fixed at a set-point per
scenario, ``p = setpoint · kW``, ``setpoint`` drawn per unit from ``STORAGE_SETPOINT`` (positive:
discharging), ``q = 0``. ``p_avail_w`` of a storage unit is its set-point: it has nothing to
curtail.

Dispatchable generators (``Generator.dg_<load>``, at every 4th customer, ``variants/dg.py``) are
set per scenario to ``p = setpoint · a · P^max``, ``setpoint`` drawn per unit from
``DG_SETPOINT``, ``q = 0``: the power flow's operating point; the OPF dispatches them within
``0 <= p, q <= a · P^max``. The availability ``a`` (0: out of service), one per unit and scenario
from ``DG_AVAILABILITY`` (a fixed value draws nothing); ``p_avail_w`` of a DG is ``a · P^max``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import PV, DGProfile, PVProfile, StorageProfile
from gridfm_datakit_distr.config.value_spec import Fixed, ValueSpec
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
    """Largest |q| in the OPF, as a fraction of ``s_rated_va``."""
    reactive_threshold: float
    """No reactive power in the OPF below this available power (fraction of ``s_rated_va``)."""

    def q_max_var(self, p_avail_w: float) -> float:
        """The largest |q| (var) the OPF may dispatch at an available power (W)."""
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
        )
        for load, kwp in sorted(p.pv_kwp.items())
    )


@dataclass(frozen=True, slots=True, eq=False)
class PVDraw:
    """One scenario's PV setpoints, ordered as the units."""

    hour: float
    cloudiness: float
    p_avail_w: NDArray[np.float64]
    p_w: NDArray[np.float64]
    """Injected active power (generation positive)."""
    q_var: NDArray[np.float64]
    """Injected reactive power (negative: absorbed)."""


class PVOutputs:
    """The outputs of a variant's PV units, scenario by scenario."""

    def __init__(self, units: Sequence[PVUnit], spec: PV, profile: PVProfile) -> None:
        self.units = tuple(units)
        self.spec = spec
        self.profile = profile
        self._kwp = np.array([u.kwp for u in self.units], dtype=np.float64)
        self._rated = np.array([u.s_rated_va for u in self.units], dtype=np.float64)
        self._tan = np.tan(np.arccos([u.power_factor for u in self.units]))

    def draw(self, seed: int, net: int, k: int, s: int, time_s: int | None) -> PVDraw:
        """Setpoints of scenario ``s``; ``time_s``: the load profile's time, if it has one."""
        if time_s is None:
            hour = float(scenario_rng(seed, net, k, s, ScenarioStream.PROFILE_TIME).uniform(0, 24))
        else:
            hour = (int(time_s) % DAY_S) / 3600.0
        rng = scenario_rng(seed, net, k, s, ScenarioStream.PV_OUTPUT)
        cloudiness = float(self.profile.cloudiness.sample(rng))
        spec = self.profile.availability
        if isinstance(spec, Fixed):
            availability: float | NDArray[np.float64] = spec.value
        else:
            rng = scenario_rng(seed, net, k, s, ScenarioStream.PV_AVAILABILITY)
            availability = np.array([float(spec.sample(rng)) for _ in self.units])
        p_avail = 1e3 * self._kwp * clear_sky(hour) * cloud_factor(cloudiness) * availability
        p = np.minimum(p_avail, self._rated)  # inverter clipping when kVA < kWp
        q_limit = np.sqrt(np.maximum(self._rated**2 - p**2, 0.0))
        q = -np.minimum(p * self._tan, q_limit)
        return PVDraw(hour, cloudiness, p_avail, p, q)


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
