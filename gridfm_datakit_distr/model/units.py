"""Per-unit conversions with explicit bases (SI internally; per unit only in exported views).

Three-phase conventions: ``s_base_va`` is the three-phase apparent-power base, ``v_ll_v`` the
line-to-line voltage base. Then ``v_ln = v_ll / sqrt(3)``, ``i_base = s / (sqrt(3) v_ll)`` and
``z_base = v_ll^2 / s``, so that ``z_base * i_base = v_ln`` (Ohm's law holds in per unit) and the
three-phase power of a balanced set, ``3 v i*``, is ``v_pu i_pu*`` per unit. Voltages to neutral
or ground are converted with ``v_ln``. A split-phase zone (0.12 kV halves) has ``v_ln = 0.12 kV``
and the three-phase-equivalent ``v_ll = 0.208 kV``, as ``Bus.kVBase`` in OpenDSS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

SQRT3 = math.sqrt(3.0)

Quantity = Literal["voltage", "current", "impedance", "admittance", "power"]
Value = TypeVar("Value", float, complex, NDArray[np.float64], NDArray[np.complex128])


@dataclass(frozen=True, slots=True)
class Bases:
    """Per-unit bases of one voltage zone."""

    s_base_va: float
    v_ll_v: float

    def __post_init__(self) -> None:
        if not (self.s_base_va > 0 and self.v_ll_v > 0):
            raise ValueError(f"bases must be positive, got S={self.s_base_va}, V={self.v_ll_v}")

    @classmethod
    def from_ln(cls, s_base_va: float, v_ln_v: float) -> Bases:
        return cls(s_base_va, v_ln_v * SQRT3)

    @property
    def v_ln_v(self) -> float:
        return self.v_ll_v / SQRT3

    @property
    def i_base_a(self) -> float:
        return self.s_base_va / (SQRT3 * self.v_ll_v)

    @property
    def z_base_ohm(self) -> float:
        return self.v_ll_v**2 / self.s_base_va

    @property
    def y_base_s(self) -> float:
        return 1.0 / self.z_base_ohm

    def base(self, quantity: Quantity) -> float:
        """Base of ``quantity`` in SI: V (to neutral), A, Ω, S, VA (three-phase)."""
        match quantity:
            case "voltage":
                return self.v_ln_v
            case "current":
                return self.i_base_a
            case "impedance":
                return self.z_base_ohm
            case "admittance":
                return self.y_base_s
            case "power":
                return self.s_base_va

    def to_pu(self, value: Value, quantity: Quantity) -> Value:
        return value / self.base(quantity)

    def from_pu(self, value: Value, quantity: Quantity) -> Value:
        return value * self.base(quantity)
