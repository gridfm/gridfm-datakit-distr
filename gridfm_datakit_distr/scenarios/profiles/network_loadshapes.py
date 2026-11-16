"""Profiles from the loadshapes of the source model.

``network_loadshapes`` and ``edp_smart_meter`` both read the loads' loadshapes through the engine:
the EDP masters attach each meter's file to its load (``Load.Pn daily=Pn``, ``Loadshape.Pn
PQCSVFile=PQ_csv/Pn.csv``), so the meter data are the loadshapes.

Series follow the engine, read from the network compiled by its recipe without the loadshape-mean
base loads, which replace the base kW the multipliers scale: the daily shape, else the yearly one;
``UseActual``: multipliers are kW and kvar, else they scale the base kW and kvar; without a Q curve,
Q keeps the base kvar/kW ratio. Point ``k`` (1-based) applies at ``k · interval``. Shapes of one
point are constants, not profiles. The profile window is the duration of the shortest multi-point
shape, so every mapped customer has a measured value at every time drawn; customers without a
multi-point shape keep a constant (their one point or their nominal value) and are reported
(:class:`Coverage`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.compile import compile_network
from gridfm_datakit_distr.scenarios.profiles.base import Customer, LoadDraw, ProfileError
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng


@dataclass(frozen=True, slots=True)
class Coverage:
    """Which customers have a profile (multi-point loadshape)."""

    n_customers: int
    n_mapped: int
    constant: tuple[str, ...]
    """Customers with a one-point loadshape."""
    unmapped: tuple[str, ...]
    """Customers without a loadshape."""

    @property
    def share(self) -> float:
        return self.n_mapped / self.n_customers if self.n_customers else 0.0


@dataclass(frozen=True, slots=True, eq=False)
class LoadshapeSeries:
    """The loads' profiles of a network over the common window."""

    interval_s: int
    names: tuple[str, ...]
    """Loads with a multi-point shape, ordered by name."""
    p_w: NDArray[np.float64]
    """``(len(names), n_times)``."""
    q_var: NDArray[np.float64]
    constants: dict[str, tuple[float, float]]
    """Loads with a one-point shape -> (P W, Q var)."""

    @property
    def times_s(self) -> NDArray[np.int64]:
        return self.interval_s * np.arange(1, self.p_w.shape[1] + 1, dtype=np.int64)

    def at(self, time_s: int) -> dict[str, tuple[float, float]]:
        """(P, Q) of every profiled load at a time of the window."""
        k, rest = divmod(int(time_s), self.interval_s)
        if rest or not 1 <= k <= self.p_w.shape[1]:
            raise ProfileError(f"time {time_s} s is not a point of the profile window")
        return {
            n: (float(self.p_w[c, k - 1]), float(self.q_var[c, k - 1]))
            for c, n in enumerate(self.names)
        }

    def coverage(self, customers: Sequence[str]) -> Coverage:
        profiled = set(self.names)
        constant = tuple(c for c in customers if c in self.constants)
        unmapped = tuple(c for c in customers if c not in profiled and c not in self.constants)
        return Coverage(len(customers), sum(c in profiled for c in customers), constant, unmapped)


def read_series(network: Network, root: Path | None = None) -> LoadshapeSeries:
    """Compile ``network`` without loadshape-mean base loads and read its loads' loadshapes."""
    ctx = DssContext.new()
    compile_network(ctx, network.model_copy(update={"base_loads": "as_is"}), root)
    api = ctx.api
    loads: list[tuple[str, float, float, str]] = []
    i = api.Loads.First()
    while i:
        shape = str(api.Loads.Daily()) or str(api.Loads.Yearly())
        if api.CktElement.Enabled() and shape:
            kw, kvar = float(api.Loads.kW()), float(api.Loads.kvar())
            loads.append((f"Load.{api.Loads.Name()}", kw, kvar, shape))
        i = api.Loads.Next()
    if not loads:
        raise ProfileError(f"{network.id}: no load has a daily or yearly loadshape")
    profiles: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    constants: dict[str, tuple[float, float]] = {}
    intervals = set()
    for name, kw, kvar, shape in sorted(loads):
        api.LoadShape.Name(shape)
        mult = np.asarray(api.LoadShape.PMult(), dtype=np.float64)
        actual = bool(api.LoadShape.UseActual())
        has_q = bool(ctx.command(f"? LoadShape.{shape}.qmult").strip())
        interval = float(api.LoadShape.SInterval())
        p = 1e3 * mult * (1.0 if actual else kw)
        if has_q:
            qmult = np.asarray(api.LoadShape.QMult(), dtype=np.float64)
            if qmult.size != mult.size:
                raise ProfileError(f"LoadShape.{shape}: {qmult.size} Q points, {mult.size} P")
            q = 1e3 * qmult * (1.0 if actual else kvar)
        else:
            q = p * (kvar / kw if kw else 0.0)
        if mult.size == 1:
            constants[name] = (float(p[0]), float(q[0]))
            continue
        if interval <= 0 or interval != round(interval):
            raise ProfileError(f"LoadShape.{shape}: interval {interval} s is not a whole second")
        intervals.add(int(interval))
        profiles[name] = (p, q)
    if len(intervals) > 1:
        raise ProfileError(
            f"{network.id}: loadshapes with different intervals {sorted(intervals)} s"
        )
    if not profiles:
        raise ProfileError(f"{network.id}: no load has a loadshape of more than one point")
    n = min(p.size for p, _ in profiles.values())
    names = tuple(sorted(profiles))
    return LoadshapeSeries(
        interval_s=intervals.pop(),
        names=names,
        p_w=np.array([profiles[c][0][:n] for c in names]),
        q_var=np.array([profiles[c][1][:n] for c in names]),
        constants=constants,
    )


class LoadshapeProvider:
    """Draws a profile time per scenario (or per customer) uniformly over the window."""

    def __init__(
        self,
        customers: Sequence[Customer],
        series: LoadshapeSeries,
        *,
        same_timestamp: bool = True,
        name: str = "network_loadshapes",
    ) -> None:
        self.name = name
        self.customers = tuple(customers)
        self.series = series
        self.same_timestamp = same_timestamp
        elements = [c.element for c in self.customers]
        self.coverage = series.coverage(elements)
        row = {n: r for r, n in enumerate(series.names)}
        self._rows = np.array([row.get(e, -1) for e in elements])
        fixed = [
            series.constants.get(c.element, (c.p_w, c.q_var)) for c in self.customers
        ]  # one-point shapes, else nominal values
        self._fixed_p = np.array([f[0] for f in fixed])
        self._fixed_q = np.array([f[1] for f in fixed])

    def draw(self, seed: int, net: int, k: int, s: int) -> LoadDraw:
        # one time for all on PROFILE_TIME; per-customer times on their own stream, since the PV
        # hour of a scenario without a common time is drawn from PROFILE_TIME
        stream = (
            ScenarioStream.PROFILE_TIME if self.same_timestamp else ScenarioStream.CUSTOMER_TIMES
        )
        rng = scenario_rng(seed, net, k, s, stream)
        n_times = self.series.p_w.shape[1]
        mapped = self._rows >= 0
        p, q = self._fixed_p.copy(), self._fixed_q.copy()
        if self.same_timestamp:
            t = int(rng.integers(n_times))
            p[mapped] = self.series.p_w[self._rows[mapped], t]
            q[mapped] = self.series.q_var[self._rows[mapped], t]
            return LoadDraw(p_w=p, q_var=q, time_s=int(self.series.times_s[t]))
        ts = rng.integers(n_times, size=len(self.customers))
        p[mapped] = self.series.p_w[self._rows[mapped], ts[mapped]]
        q[mapped] = self.series.q_var[self._rows[mapped], ts[mapped]]
        return LoadDraw(p_w=p, q_var=q, customer_times_s=self.series.times_s[ts])
