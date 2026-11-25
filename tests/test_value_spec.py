"""Tests for value specs.

Adapted from gridfm-datakit's ``tests/test_value_spec.py`` at commit 1b974c0 (Apache License 2.0,
https://github.com/gridfm/gridfm-datakit): same cases against the vendored module, plus the
path-less parser and the round trip of ``to_config`` added here.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pytest
from scipy import stats

from gridfm_datakit_distr.config.value_spec import (
    NON_NEGATIVE,
    POSITIVE,
    Choice,
    Fixed,
    Floor,
    LogUniform,
    Normal,
    Uniform,
    ValueSpec,
    parse_spec,
    parse_value_spec,
)

PATH = "variants.admittance.line_length_scale"
MODULE = "gridfm_datakit_distr.config.value_spec"


def _draws(spec: ValueSpec, seed: int, n: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([spec.sample(rng) for _ in range(n)])


class TestParse:
    def test_a_number_is_fixed(self) -> None:
        spec = parse_value_spec(3, PATH)
        assert spec == Fixed(3.0)
        assert spec.support() == (3.0, 3.0)
        assert spec.sample(np.random.default_rng(0)) == 3.0

    def test_normal(self) -> None:
        spec = parse_value_spec({"distribution": "normal", "mean": 1, "std": 2}, PATH)
        assert isinstance(spec, Normal)
        assert spec.support() == (-11.0, 13.0)

    def test_bounded_normal(self) -> None:
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0, "std": 1, "min": -0.5, "max": 2}, PATH
        )
        assert spec.support() == (-0.5, 2.0)

    def test_normal_with_a_floor(self) -> None:
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0.1, "std": 0.05}, PATH, POSITIVE
        )
        assert spec.support() == (0.0, pytest.approx(0.4))

    def test_one_sided_normal_support(self) -> None:
        spec = parse_value_spec({"distribution": "normal", "mean": 10, "std": 1, "min": 15.9}, PATH)
        assert spec.support() == (15.9, pytest.approx(21.9))
        spec = parse_value_spec({"distribution": "normal", "mean": 0, "std": 1, "max": -5.9}, PATH)
        assert spec.support() == (pytest.approx(-11.9), -5.9)

    def test_uniform(self) -> None:
        spec = parse_value_spec({"distribution": "uniform", "low": 1, "high": 4}, PATH)
        assert isinstance(spec, Uniform)
        assert spec.support() == (1.0, 4.0)

    def test_log_uniform(self) -> None:
        spec = parse_value_spec({"distribution": "log_uniform", "low": 0.5, "high": 2}, PATH)
        assert spec == LogUniform(0.5, 2.0)
        assert spec.support() == (0.5, 2.0)

    def test_choice(self) -> None:
        spec = parse_value_spec(
            {"distribution": "choice", "values": [3, 1, 2], "weights": [1, 1, 2]}, PATH
        )
        assert isinstance(spec, Choice)
        assert spec.support() == (1, 3)
        assert spec.probabilities == (0.25, 0.25, 0.5)

    def test_path_less_parser_gives_the_same_specs_and_bare_messages(self) -> None:
        raw = {"distribution": "uniform", "low": 1, "high": 4}
        assert parse_spec(raw) == parse_value_spec(raw, PATH)
        with pytest.raises(ValueError, match=r"^low must be below high, got 4.0 and 1.0$"):
            parse_spec({"distribution": "uniform", "low": 4, "high": 1})


class TestRejects:
    def _raises(self, spec: Any, *names: str, floor: Floor | None = None) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: ") as info:
            parse_value_spec(spec, PATH, floor)
        message = str(info.value)
        assert message.startswith(f"{PATH}: "), message
        for name in names:
            assert name in message, message

    def test_sigma_is_not_a_key(self) -> None:
        self._raises({"distribution": "normal", "mean": 0, "std": 1, "sigma": 1}, "sigma")

    def test_missing_distribution(self) -> None:
        self._raises({"mean": 0, "std": 1}, "normal", "uniform", "choice")

    def test_a_list_distribution(self) -> None:
        self._raises(
            {"distribution": ["uniform"], "low": 1, "high": 2}, "normal", "uniform", "choice"
        )

    def test_unknown_distribution(self) -> None:
        self._raises(
            {"distribution": "lognormal", "mean": 0, "std": 1},
            "lognormal",
            "normal",
            "uniform",
            "choice",
        )

    @pytest.mark.parametrize(
        "spec",
        [
            {"distribution": "normal", "mean": 0, "std": 0},
            {"distribution": "uniform", "low": 1, "high": 1},
            {"distribution": "log_uniform", "low": 0, "high": 1},
            {"distribution": "log_uniform", "low": 2, "high": 1},
            {"distribution": "normal", "mean": 0, "std": 1, "min": 1, "max": -1},
            {"distribution": "choice", "values": []},
            {"distribution": "choice", "values": [1, 2], "weights": [1]},
            {"distribution": "choice", "values": [1, 2], "weights": [1, -1]},
            {"distribution": "choice", "values": [1, 2], "weights": [0, 0]},
            {"distribution": "normal", "mean": True, "std": 1},
            True,
            "1.0",
        ],
    )
    def test_invalid_parameters(self, spec: Any) -> None:
        self._raises(spec)

    def test_missing_parameter(self) -> None:
        self._raises({"distribution": "uniform", "low": 0}, "high")

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_numbers(self, value: float) -> None:
        self._raises(value, "finite")
        self._raises({"distribution": "normal", "mean": 0, "std": value}, "std")


class TestSample:
    def test_bounded_normal_redraws_rather_than_clips(self) -> None:
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0, "std": 1, "min": -0.5, "max": 0.5}, PATH
        )
        draws = _draws(spec, 0)
        assert draws.min() > -0.5
        assert draws.max() < 0.5
        assert np.array_equal(draws, _draws(spec, 0))
        assert not np.array_equal(draws, _draws(spec, 1))

    @pytest.mark.parametrize(
        "spec",
        [
            {"mean": 0, "std": 1, "min": 5.9},
            {"mean": 0, "std": 1, "max": -5.9},
            {"mean": 10, "std": 1, "min": 15.9},
        ],
    )
    def test_one_sided_tail_draws_stay_inside_the_support(self, spec: dict[str, float]) -> None:
        parsed = parse_value_spec({"distribution": "normal", **spec}, PATH)
        low, high = parsed.support()
        draws = _draws(parsed, 0)
        assert ((low <= draws) & (draws <= high)).all()

    def test_an_unbounded_normal_is_drawn_inside_its_support(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[float, float]] = []

        class Stub:
            def rvs(self, a: float, b: float, **kwargs: Any) -> float:
                calls.append((a, b))
                return 0.0

        monkeypatch.setattr(f"{MODULE}.truncnorm", Stub())
        spec = parse_value_spec({"distribution": "normal", "mean": 2, "std": 1}, PATH)
        spec.sample(np.random.default_rng(0))
        assert calls == [(-6.0, 6.0)]

    def test_normal_draws_are_floats(self) -> None:
        spec = parse_value_spec({"distribution": "normal", "mean": 0, "std": 1}, PATH)
        assert type(spec.sample(np.random.default_rng(0))) is float

    def test_uniform_stays_in_its_half_open_interval(self) -> None:
        spec = parse_value_spec({"distribution": "uniform", "low": 2, "high": 3}, PATH)
        draws = _draws(spec, 0)
        assert draws.min() >= 2
        assert draws.max() < 3
        assert type(spec.sample(np.random.default_rng(0))) is float

    def test_log_uniform_is_log_uniform_with_median_one_on_reciprocal_bounds(self) -> None:
        spec = LogUniform(1 / 2.64, 2.64)
        draws = _draws(spec, 0)
        assert stats.kstest(draws, stats.loguniform(1 / 2.64, 2.64).cdf).pvalue > 1e-3
        assert abs(np.median(draws) - 1.0) < 0.05  # median sqrt(low · high) = 1
        assert draws.min() >= 1 / 2.64
        assert draws.max() <= 2.64
        assert type(spec.sample(np.random.default_rng(0))) is float

    def test_choice_never_draws_a_zero_weight(self) -> None:
        spec = parse_value_spec(
            {"distribution": "choice", "values": ["a", "b", "c"], "weights": [1, 0, 3]}, PATH
        )
        rng = np.random.default_rng(0)
        assert {spec.sample(rng) for _ in range(1_000)} == {"a", "c"}


class TestFloor:
    def test_fixed_zero(self) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec(0, PATH, POSITIVE)
        assert parse_value_spec(0, PATH, NON_NEGATIVE) == Fixed(0.0)

    def test_log_uniform_low_must_be_admitted(self) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: every value must be >= 1"):
            parse_value_spec({"distribution": "log_uniform", "low": 0.5, "high": 2}, PATH,
                             Floor(1.0, strict=False))  # fmt: skip

    def test_uniform_low_must_be_admitted(self) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec({"distribution": "uniform", "low": -1, "high": 1}, PATH, POSITIVE)

    def test_normal_is_truncated_at_the_floor(self) -> None:
        spec = parse_value_spec(
            {"distribution": "normal", "mean": 0.1, "std": 0.05}, PATH, POSITIVE
        )
        assert (_draws(spec, 0) > 0).all()

    def test_normal_redraws_a_value_on_a_strict_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Stub:
            def __init__(self) -> None:
                self.values = iter([0.0, 0.3])

            def rvs(self, *args: Any, **kwargs: Any) -> float:
                return next(self.values)

        spec = {"distribution": "normal", "mean": 0.1, "std": 0.05}
        rng = np.random.default_rng(0)
        monkeypatch.setattr(f"{MODULE}.truncnorm", Stub())
        positive = parse_value_spec(spec, PATH, POSITIVE)
        assert isinstance(positive, Normal)
        assert positive.strict
        assert positive.sample(rng) == 0.3
        monkeypatch.setattr(f"{MODULE}.truncnorm", Stub())
        assert parse_value_spec(spec, PATH, NON_NEGATIVE).sample(rng) == 0.0
        bounded = parse_value_spec({**spec, "min": 0.05}, PATH, POSITIVE)
        assert isinstance(bounded, Normal)
        assert not bounded.strict

    def test_a_strict_floor_too_close_to_max_raises(self) -> None:
        spec = {"distribution": "normal", "mean": 0, "std": 1, "max": 1e-200}
        positive = parse_value_spec(spec, PATH, POSITIVE)
        with pytest.raises(ValueError, match=r"no draw above 0\.0"):
            positive.sample(np.random.default_rng(0))

    def test_normal_far_below_the_floor(self) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec({"distribution": "normal", "mean": -10, "std": 1}, PATH, POSITIVE)

    def test_choice_of_a_string(self) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(PATH)}: "):
            parse_value_spec({"distribution": "choice", "values": [1, "a"]}, PATH, POSITIVE)

    def test_admits(self) -> None:
        assert not POSITIVE.admits(0.0)
        assert POSITIVE.admits(math.ulp(0.0))
        assert NON_NEGATIVE.admits(0.0)
        assert not NON_NEGATIVE.admits(-1e-12)


class TestToConfig:
    @pytest.mark.parametrize(
        ("raw", "floor"),
        [
            (2.5, None),
            ({"distribution": "uniform", "low": 0.9, "high": 1.1}, POSITIVE),
            ({"distribution": "log_uniform", "low": 1 / 2.64, "high": 2.64}, POSITIVE),
            ({"distribution": "normal", "mean": 0.1, "std": 0.05}, POSITIVE),
            ({"distribution": "normal", "mean": 1, "std": 2, "min": -1}, None),
            ({"distribution": "normal", "mean": 1, "std": 2, "max": 3}, None),
            ({"distribution": "choice", "values": [1.0, 5.0, 20.0]}, POSITIVE),
            ({"distribution": "choice", "values": ["a", "b"], "weights": [1, 3]}, None),
        ],
    )
    def test_round_trip(self, raw: Any, floor: Floor | None) -> None:
        spec = parse_spec(raw, floor)
        again = parse_spec(spec.to_config(), floor)
        assert again == spec

    def test_forms(self) -> None:
        assert Fixed(2.0).to_config() == 2.0
        assert Uniform(1.0, 2.0).to_config() == {"distribution": "uniform", "low": 1.0, "high": 2.0}
        assert LogUniform(0.5, 2.0).to_config() == {
            "distribution": "log_uniform",
            "low": 0.5,
            "high": 2.0,
        }
        assert Normal(0.0, 1.0, lower=-1.0).to_config() == {
            "distribution": "normal",
            "mean": 0.0,
            "std": 1.0,
            "min": -1.0,
        }
        assert Choice(("a", "b"), (0.25, 0.75)).to_config() == {
            "distribution": "choice",
            "values": ["a", "b"],
            "weights": [0.25, 0.75],
        }
