"""Load profiles: the customers of a variant and what a scenario draws for them.

A provider is built for one variant and its customers (enabled loads, ordered by name) and draws
each scenario's active and reactive powers from the scenario's own streams. Profile time
is the offset from the start of the source's profile in seconds (``time_s``): no catalog source
dates its data (European LV: one unnamed day at 1-minute steps; EDP meter files: rows of P and Q
without timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.networks.prepare import PreparedNetwork


class ProfileError(RuntimeError):
    """A provider cannot draw for a network (no loadshapes, unsupported shape, ...)."""


@dataclass(frozen=True, slots=True)
class Customer:
    element: str
    """``Load.<name>``."""
    p_w: float
    """Nominal active power of the variant (W): its base kW after the catalog recipe."""
    q_var: float
    """Nominal reactive power of the variant (var)."""


@dataclass(frozen=True, slots=True, eq=False)
class LoadDraw:
    """One scenario's setpoints, ordered as the provider's customers."""

    p_w: NDArray[np.float64]
    q_var: NDArray[np.float64]
    time_s: int | None = None
    """Profile time shared by all customers (``same_timestamp``); None otherwise."""
    customer_times_s: NDArray[np.int64] | None = None
    """Profile time of each customer when they do not share one."""
    global_scale: float | None = None
    """The synthetic provider's factor on all nominal loads."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LoadDraw):
            return NotImplemented
        times_equal = (self.customer_times_s is None and other.customer_times_s is None) or (
            self.customer_times_s is not None
            and other.customer_times_s is not None
            and np.array_equal(self.customer_times_s, other.customer_times_s)
        )
        return (
            np.array_equal(self.p_w, other.p_w)
            and np.array_equal(self.q_var, other.q_var)
            and self.time_s == other.time_s
            and times_equal
            and self.global_scale == other.global_scale
        )


class ProfileProvider(Protocol):
    name: str
    customers: tuple[Customer, ...]

    def draw(self, seed: int, i: int, k: int, s: int) -> LoadDraw:
        """Setpoints of scenario ``s`` of variant ``k`` of network ``i``."""
        ...


def customers_of(prepared: PreparedNetwork) -> tuple[Customer, ...]:
    """The enabled loads of a prepared variant with their nominal powers, ordered by name."""
    api = prepared.ctx.api
    out = []
    i = api.Loads.First()
    while i:
        if api.CktElement.Enabled():
            out.append(
                Customer(
                    f"Load.{api.Loads.Name()}",
                    float(api.Loads.kW()) * 1e3,
                    float(api.Loads.kvar()) * 1e3,
                )
            )
        i = api.Loads.Next()
    return tuple(sorted(out, key=lambda c: c.element))
