"""Drawn values for config entries (value specs).

Adapted from gridfm-datakit, ``gridfm_datakit/utils/value_spec.py`` at commit 1b974c0
(https://github.com/gridfm/gridfm-datakit), Apache License 2.0. The semantics are unchanged.
Changes: typed for ``mypy --strict``; :func:`parse_spec` parses without a config path (its
messages carry none, for pydantic, which adds the location); ``to_config`` returns a spec's
config form (round trip: ``parse_spec(s.to_config(), floor) == s``).

Forms:
    number                                              fixed
    {distribution: normal, mean, std, min?, max?}       truncated normal
    {distribution: uniform, low, high}                  on [low, high)
    {distribution: choice, values, weights?}            one listed value

Out of range values are redrawn, never clipped: a normal is sampled from
``scipy.stats.truncnorm`` on its ``support()``, ``[min, max]`` intersected with
the floor, an unbounded side ending 6 std beyond the mean, or beyond the other
side's bound when that is further out. A truncation interval with no overlap with
``[mean - 6 std, mean + 6 std]`` is rejected at parse. Draws never leave the
support, so a caller validates against it.

A floor is checked at parse: fixed, uniform and choice specs are rejected when
their support fails it, a normal takes it as a lower truncation bound and redraws a
value equal to a strict one, at most ``_MAX_REDRAWS`` times: ``truncnorm`` returns
the bound itself on an interval too narrow to resolve.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import truncnorm

_NORMAL_SPAN = 6.0
_MAX_REDRAWS = 1000

_KEYS = {
    "normal": ({"mean", "std"}, {"min", "max"}),
    "uniform": ({"low", "high"}, set()),
    "choice": ({"values"}, {"weights"}),
}


@dataclass(frozen=True)
class Floor:
    """Lower bound every drawn value must admit.

    Args:
        bound: The bound.
        strict: Whether the bound itself is excluded.
    """

    bound: float
    strict: bool

    def admits(self, x: float) -> bool:
        """Whether ``x`` lies above the bound, or on it when not strict."""
        return x > self.bound if self.strict else x >= self.bound


POSITIVE = Floor(0.0, strict=True)
NON_NEGATIVE = Floor(0.0, strict=False)


class ValueSpec:
    """A value drawn once per use."""

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value: a float for numeric specs, the list element for a choice."""
        raise NotImplementedError

    def support(self) -> tuple[Any, Any]:
        """The smallest and largest value a draw can take (worst case bounds)."""
        raise NotImplementedError

    def to_config(self) -> Any:
        """The spec in its config form: a number or a mapping with a ``distribution`` key."""
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(ValueSpec):
    """A constant: every draw returns ``value``."""

    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.value

    def support(self) -> tuple[float, float]:
        return self.value, self.value

    def to_config(self) -> float:
        return self.value


@dataclass(frozen=True)
class Normal(ValueSpec):
    """A normal distribution truncated to its support.

    Args:
        mean: The mean before truncation.
        std: The standard deviation before truncation.
        lower: The lower truncation bound, ``-inf`` when unbounded.
        upper: The upper truncation bound, ``inf`` when unbounded.
        strict: Whether ``lower`` is excluded.
    """

    mean: float
    std: float
    lower: float = -math.inf
    upper: float = math.inf
    strict: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        """Draw from the truncated normal; raises ValueError if no draw lands above a strict
        ``lower``."""
        lower, upper = self.support()
        a = (lower - self.mean) / self.std
        b = (upper - self.mean) / self.std
        for _ in range(_MAX_REDRAWS):
            x = float(truncnorm.rvs(a, b, loc=self.mean, scale=self.std, random_state=rng))
            if not (self.strict and x <= self.lower):
                return x
        raise ValueError(
            f"no draw above {self.lower} in [{lower}, {upper}] after {_MAX_REDRAWS} attempts"
        )

    def support(self) -> tuple[float, float]:
        """An unbounded side lies 6 std beyond the mean or beyond the bound on the other side,
        whichever is further out."""
        span = _NORMAL_SPAN * self.std
        lower = self.lower
        if math.isinf(lower):
            lower = min(self.mean, self.upper) - span
        upper = self.upper
        if math.isinf(upper):
            upper = max(self.mean, self.lower) + span
        return lower, upper

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"distribution": "normal", "mean": self.mean, "std": self.std}
        if not math.isinf(self.lower):
            config["min"] = self.lower
        if not math.isinf(self.upper):
            config["max"] = self.upper
        return config


@dataclass(frozen=True)
class Uniform(ValueSpec):
    """A uniform distribution on ``[low, high)``."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def support(self) -> tuple[float, float]:
        return self.low, self.high

    def to_config(self) -> dict[str, Any]:
        return {"distribution": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Choice(ValueSpec):
    """One of a list of values, drawn with the given probabilities."""

    values: tuple[Any, ...]
    probabilities: tuple[float, ...]

    def sample(self, rng: np.random.Generator) -> Any:
        return self.values[rng.choice(len(self.values), p=self.probabilities)]

    def support(self) -> tuple[Any, Any]:
        return min(self.values), max(self.values)

    def to_config(self) -> dict[str, Any]:
        return {
            "distribution": "choice",
            "values": list(self.values),
            "weights": list(self.probabilities),
        }


def parse_value_spec(spec: Any, path: str, floor: Floor | None = None) -> ValueSpec:
    """Parse a config entry into a value spec.

    Args:
        spec: A number or a mapping with a ``distribution`` key.
        path: The config path named in every error.
        floor: A lower bound every value must admit.

    Raises:
        ValueError: If the entry is malformed, a parameter is invalid, or a value can fail the
            floor. The message starts with ``f"{path}: "``.
    """
    try:
        return parse_spec(spec, floor)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def parse_spec(spec: Any, floor: Floor | None = None) -> ValueSpec:
    """:func:`parse_value_spec` without a path: messages state the reason only."""
    if not isinstance(spec, Mapping):
        value = _number(spec, "value")
        _check_floor(value, floor)
        return Fixed(value)

    distribution = spec.get("distribution")
    if not isinstance(distribution, str) or distribution not in _KEYS:
        raise ValueError(f"distribution must be one of {sorted(_KEYS)}, got {distribution!r}")
    required, optional = _KEYS[distribution]
    unknown = sorted(set(spec) - required - optional - {"distribution"})
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} for {distribution}")
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"missing key(s) {missing} for {distribution}")

    if distribution == "normal":
        return _parse_normal(spec, floor)
    if distribution == "uniform":
        return _parse_uniform(spec, floor)
    return _parse_choice(spec, floor)


def _parse_normal(spec: Mapping[str, Any], floor: Floor | None) -> Normal:
    mean = _number(spec["mean"], "mean")
    std = _number(spec["std"], "std")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    lower = _number(spec["min"], "min") if "min" in spec else -math.inf
    upper = _number(spec["max"], "max") if "max" in spec else math.inf
    if lower >= upper:
        raise ValueError(f"min must be below max, got {lower} and {upper}")
    if floor is not None:
        lower = max(lower, floor.bound)
    span = _NORMAL_SPAN * std
    if lower >= min(upper, mean + span) or upper <= max(lower, mean - span):
        raise ValueError(
            f"truncation interval [{lower}, {upper}] misses "
            f"[{mean - span}, {mean + span}], mean +/- {_NORMAL_SPAN:g} std"
        )
    strict = floor is not None and floor.strict and lower == floor.bound
    return Normal(mean, std, lower, upper, strict)


def _parse_uniform(spec: Mapping[str, Any], floor: Floor | None) -> Uniform:
    low = _number(spec["low"], "low")
    high = _number(spec["high"], "high")
    if low >= high:
        raise ValueError(f"low must be below high, got {low} and {high}")
    _check_floor(low, floor)
    return Uniform(low, high)


def _parse_choice(spec: Mapping[str, Any], floor: Floor | None) -> Choice:
    values = spec["values"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"values must be a list, got {values!r}")
    if not values:
        raise ValueError("values must not be empty")
    if floor is not None:
        for value in values:
            _check_floor(_number(value, "values"), floor)

    weights = spec.get("weights")
    if weights is None:
        probabilities = tuple(1.0 / len(values) for _ in values)
    else:
        if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
            raise ValueError(f"weights must be a list, got {weights!r}")
        if len(weights) != len(values):
            raise ValueError(f"weights has {len(weights)} entries, values has {len(values)}")
        numeric = [_number(w, "weights") for w in weights]
        if any(w < 0 for w in numeric):
            raise ValueError(f"weights must be non-negative, got {numeric}")
        total = sum(numeric)
        if total <= 0:
            raise ValueError("weights must have a positive sum")
        probabilities = tuple(w / total for w in numeric)
    return Choice(tuple(values), probabilities)


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def _check_floor(value: float, floor: Floor | None) -> None:
    if floor is not None and not floor.admits(value):
        relation = ">" if floor.strict else ">="
        raise ValueError(f"every value must be {relation} {floor.bound}, got {value}")


__all__ = [
    "NON_NEGATIVE",
    "POSITIVE",
    "Choice",
    "Fixed",
    "Floor",
    "Normal",
    "Uniform",
    "ValueSpec",
    "parse_spec",
    "parse_value_spec",
]
