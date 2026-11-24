"""PVGIS PV series: parsing, cache and checksums, draws in local time, outputs, config, CLI.

Series are built here in PVGIS's JSON layout from made-up numbers (no PVGIS data in git):
``P`` of record ``i`` is ``i`` W unless stated, so a draw names its record.
"""

from __future__ import annotations

import copy
import io
import json
import re
import urllib.error
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from gridfm_datakit_distr.config.load import ConfigError, stored_config, validate_config
from gridfm_datakit_distr.config.schema import PV, PVGIS, PVGISLocation, PVProfile
from gridfm_datakit_distr.networks.catalog import Catalog, load_catalog
from gridfm_datakit_distr.scenarios import pvgis
from gridfm_datakit_distr.scenarios.der import PVOutputs, PVUnit
from gridfm_datakit_distr.scenarios.pvgis import (
    PVGISError,
    PVSeries,
    draw_record,
    fetch_series,
    local_days,
    missing_series,
    read_series,
    request_url,
    series_path,
)

MANCHESTER = PVGISLocation(lat=53.48, lon=-2.24, timezone="Europe/London")


def document(
    first: int = 2020,
    last: int = 2020,
    p: Callable[[int], float] = float,
    minute: int = 11,
    **meteo: Any,
) -> dict[str, Any]:
    """A ``seriescalc`` response with PV power for every hour of ``first``-``last`` (UTC)."""
    start = datetime(first, 1, 1, tzinfo=UTC)
    n = (datetime(last + 1, 1, 1, tzinfo=UTC) - start) // timedelta(hours=1)
    hourly = [
        {
            "time": (start + timedelta(hours=i)).strftime("%Y%m%d:%H") + f"{minute:02d}",
            "P": p(i),
            "G(i)": 0.0,
            "H_sun": 0.0,
            "T2m": 10.0,
            "WS10m": 1.0,
            "Int": 0.0,
        }
        for i in range(n)
    ]
    return {
        "inputs": {
            "location": {"latitude": 53.48, "longitude": -2.24, "elevation": 50.0},
            "meteo_data": {
                "radiation_db": "PVGIS-SARAH3",
                "meteo_db": "ERA5",
                "year_min": first,
                "year_max": last,
                "use_horizon": True,
                "horizon_db": None,
                "horizon_data": "DEM-calculated",
                **meteo,
            },
            "mounting_system": {
                "fixed": {
                    "slope": {"value": 40, "optimal": True},
                    "azimuth": {"value": -5, "optimal": True},
                    "type": "free-standing",
                }
            },
            "pv_module": {"technology": "c-Si", "peak_power": 1.0, "system_loss": 14.0},
        },
        "outputs": {"hourly": hourly},
        "meta": {},
    }


def encode(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc).encode()


def series(doc: dict[str, Any] | None = None, tz: str = "UTC") -> PVSeries:
    parsed = pvgis._series(doc or document(), "0" * 64, "test")
    return PVSeries(**{**_fields(parsed), "tz": ZoneInfo(tz)})


def _fields(s: PVSeries) -> dict[str, Any]:
    return {name: getattr(s, name) for name in PVSeries.__dataclass_fields__}


class FixedRng:
    """Stands for a generator whose next integer is known."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.bounds: list[int] = []

    def integers(self, n: int) -> int:
        self.bounds.append(n)
        assert 0 <= self.value < n
        return self.value


def utc_hour(s: PVSeries, i: int) -> datetime:
    return datetime.fromtimestamp(s.start_s + 3600 * i, UTC)


# ------------------------------------------------------------------------------ parsing


def test_a_series_is_read_with_its_inputs() -> None:
    s = series(document(2019, 2020))  # 2020 is a leap year
    assert len(s.p_w) == 8760 + 8784
    assert s.start_s == int(datetime(2019, 1, 1, tzinfo=UTC).timestamp())
    assert np.array_equal(s.p_w, np.arange(len(s.p_w), dtype=float))
    assert (s.radiation_db, s.meteo_db, s.slope_deg, s.azimuth_deg) == (
        "PVGIS-SARAH3", "ERA5", 40.0, -5.0,
    )  # fmt: skip
    assert utc_hour(s, 8760 + 59 * 24) == datetime(2020, 2, 29, tzinfo=UTC)


def test_a_record_stands_for_the_hour_starting_at_its_hh00() -> None:
    for minute in (0, 10, 30, 59):
        s = series(document(minute=minute))
        assert utc_hour(s, 13) == datetime(2020, 1, 1, 13, tzinfo=UTC)


def _drop(i: int) -> Callable[[dict[str, Any]], None]:
    return lambda d: d["outputs"]["hourly"].pop(i)


def _repeat(i: int) -> Callable[[dict[str, Any]], None]:
    return lambda d: d["outputs"]["hourly"].insert(i, d["outputs"]["hourly"][i])


def _set(key: str, value: Any, i: int = 5) -> Callable[[dict[str, Any]], None]:
    return lambda d: d["outputs"]["hourly"][i].__setitem__(key, value)


def _year(value: int) -> Callable[[dict[str, Any]], None]:
    return lambda d: d["inputs"]["meteo_data"].__setitem__("year_min", value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_drop(100), "8783 hourly records where 2020-2020 has 8784; record 100"),
        (_drop(8783), "record 8783 is nothing, expected 2020-12-31T23"),
        (_repeat(7), "record 8 is 2020-01-01T07, expected 2020-01-01T08"),
        (_set("time", "20200101:0560"), "not YYYYMMDD:HHMM"),
        (_set("time", "2020-01-01T05"), "not YYYYMMDD:HHMM"),
        (_set("time", "20200230:0511"), "invalid time"),
        (_set("P", -1.0), "record 5 (20200101:0511) has P = -1.0"),
        (_set("P", None), "has P = nan"),
        (_set("P", "x"), "not a PVGIS seriescalc response"),
        (lambda d: d["outputs"]["hourly"][3].pop("P"), "KeyError: 'P'"),
        (lambda d: d.pop("inputs"), "KeyError: 'inputs'"),
        (_year(2019), "where 2019-2020 has 17544; record 0 is 2020-01-01T00"),
    ],
)
def test_a_malformed_series_is_refused(
    change: Callable[[dict[str, Any]], None], message: str
) -> None:
    doc = document()
    change(doc)
    with pytest.raises(PVGISError, match=re.escape(message)):
        pvgis._series(doc, "0" * 64, "test")


def test_a_response_that_is_not_json_is_refused() -> None:
    with pytest.raises(PVGISError, match="not a JSON document"):
        pvgis._document(b"<html>busy</html>", "test")
    with pytest.raises(PVGISError, match="a JSON list"):
        pvgis._document(b"[]", "test")


# ------------------------------------------------------------------------------ requests


def test_the_request_names_every_parameter() -> None:
    url = request_url(PVGIS(), MANCHESTER)
    assert url == (
        "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat=53.48&lon=-2.24&startyear=2005"
        "&endyear=2023&pvcalculation=1&peakpower=1&loss=14.0&pvtechchoice=crystSi"
        "&mountingplace=free&optimalangles=1&outputformat=json"
    )
    fixed = PVGIS(optimalangles=False, angle=35, aspect=-90, mountingplace="building")
    assert "&mountingplace=building&angle=35&aspect=-90&outputformat" in request_url(
        fixed, MANCHESTER
    )
    assert series_path(url, Path("/c")).parent == Path("/c/pvgis")
    assert series_path(url, Path("/c")) != series_path(request_url(fixed, MANCHESTER), Path("/c"))


class FakeHTTP:
    """``urlopen`` answering with a list of statuses, then the body."""

    def __init__(self, statuses: list[int], body: bytes, error_body: bytes = b"") -> None:
        self.statuses = list(statuses)
        self.body = body
        self.error_body = error_body
        self.calls: list[str] = []

    def __call__(self, request: Any, timeout: float) -> Any:
        self.calls.append(request.full_url)
        assert request.get_header("User-agent").startswith("gridfm-datakit-distr/")
        if self.statuses:
            code = self.statuses.pop(0)
            raise urllib.error.HTTPError(
                request.full_url, code, "x", {}, io.BytesIO(self.error_body)
            )
        return io.BytesIO(self.body)


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeHTTP]:
    sleeps: list[float] = []
    monkeypatch.setattr(pvgis.time, "sleep", sleeps.append)

    def install(statuses: list[int], body: bytes = b"", error_body: bytes = b"") -> FakeHTTP:
        fake = FakeHTTP(statuses, body, error_body)
        monkeypatch.setattr(pvgis.urllib.request, "urlopen", fake)
        return fake

    return install


def one_year() -> PVGIS:
    return PVGIS(startyear=2020, endyear=2020)


def test_fetch_caches_the_series_with_its_checksum(
    tmp_path: Path, http: Callable[..., FakeHTTP]
) -> None:
    body = encode(document())
    fake = http([429, 529], body)  # retried
    path, s, downloaded = fetch_series(one_year(), MANCHESTER, tmp_path)
    assert downloaded
    assert len(fake.calls) == 3
    assert path.read_bytes() == body
    listed = path.with_name(path.name + ".sha256").read_text()
    assert listed == f"{s.sha256}  {path.name}\n"
    assert not list(path.parent.glob(".*.tmp"))
    again = http([], b"never asked")
    assert fetch_series(one_year(), MANCHESTER, tmp_path)[2] is False
    assert again.calls == []


def test_a_changed_cached_series_is_an_error_never_a_silent_download(
    tmp_path: Path, http: Callable[..., FakeHTTP]
) -> None:
    http([], encode(document()))
    path, _, _ = fetch_series(one_year(), MANCHESTER, tmp_path)
    data = path.read_bytes()
    path.write_bytes(data.replace(b'"P": 5.0', b'"P": 6.0'))
    fake = http([], encode(document()))
    with pytest.raises(PVGISError, match="changed since it was downloaded"):
        fetch_series(one_year(), MANCHESTER, tmp_path)
    with pytest.raises(PVGISError, match="changed since it was downloaded"):
        read_series(path)
    assert fake.calls == []
    path.with_name(path.name + ".sha256").unlink()
    with pytest.raises(PVGISError, match="no SHA-256 recorded"):
        read_series(path)
    _, s, downloaded = fetch_series(one_year(), MANCHESTER, tmp_path, force=True)
    assert downloaded
    assert read_series(path).sha256 == s.sha256


@pytest.mark.parametrize(
    ("listed", "message"),
    [
        ("not a checksum line\n", "expected '<sha256>  <path>'"),
        (f"{'0' * 64}  other.json\n", "no SHA-256 recorded"),
        ("", "no SHA-256 recorded"),
    ],
)
def test_a_checksum_file_must_list_the_series(
    tmp_path: Path, http: Callable[..., FakeHTTP], listed: str, message: str
) -> None:
    http([], encode(document()))
    path, _, _ = fetch_series(one_year(), MANCHESTER, tmp_path)
    path.with_name(path.name + ".sha256").write_text(listed)
    with pytest.raises(PVGISError, match=message):
        read_series(path)


def test_rate_limits_are_retried_a_bounded_number_of_times(
    tmp_path: Path, http: Callable[..., FakeHTTP]
) -> None:
    fake = http([529] * 10, encode(document()))
    with pytest.raises(PVGISError, match=r"HTTP 529 \(overloaded\) 4 times; try again later"):
        fetch_series(one_year(), MANCHESTER, tmp_path)
    assert len(fake.calls) == pvgis.ATTEMPTS
    assert not (tmp_path / "pvgis").exists() or not list((tmp_path / "pvgis").iterdir())


def test_a_refused_request_says_why(tmp_path: Path, http: Callable[..., FakeHTTP]) -> None:
    error = b'{"message":"Location over the sea. Please, select another location","status":400}'
    fake = http([400], error_body=error)
    with pytest.raises(PVGISError, match="HTTP 400: Location over the sea"):
        fetch_series(one_year(), MANCHESTER, tmp_path)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["inputs"]["location"].__setitem__("latitude", 53.5), "lat 53.5"),
        (lambda d: d["inputs"]["pv_module"].__setitem__("system_loss", 10.0), "loss 10.0"),
        (
            lambda d: d["inputs"]["mounting_system"]["fixed"].__setitem__(
                "type", "building-integrated"
            ),
            "mountingplace 'building-integrated'",
        ),
    ],
)
def test_a_response_must_report_the_request(
    tmp_path: Path,
    http: Callable[..., FakeHTTP],
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    doc = document()
    change(doc)
    http([], encode(doc))
    with pytest.raises(PVGISError, match=f"reports other inputs: {message}"):
        fetch_series(one_year(), MANCHESTER, tmp_path)
    assert not series_path(request_url(one_year(), MANCHESTER), tmp_path).exists()


def test_explicit_angles_must_be_the_ones_used(
    tmp_path: Path, http: Callable[..., FakeHTTP]
) -> None:
    config = PVGIS(startyear=2020, endyear=2020, optimalangles=False, angle=35, aspect=0)
    doc = document()
    fixed = doc["inputs"]["mounting_system"]["fixed"]
    fixed["slope"] = {"value": 35, "optimal": False}
    fixed["azimuth"] = {"value": 0, "optimal": False}
    http([], encode(doc))
    assert fetch_series(config, MANCHESTER, tmp_path)[1].slope_deg == 35.0
    fixed["slope"]["value"] = 36
    http([], encode(doc))
    with pytest.raises(PVGISError, match="angle 36 \\(asked 35\\)"):
        fetch_series(config, MANCHESTER, tmp_path, force=True)


def test_missing_series_are_named_and_cached_ones_checked(
    tmp_path: Path, http: Callable[..., FakeHTTP]
) -> None:
    config = PVGIS(
        startyear=2020,
        endyear=2020,
        locations={
            "a/*": MANCHESTER,
            "b": PVGISLocation(lat=43.4, lon=-5.8, timezone="Europe/Madrid"),
        },
    )
    assert missing_series(config, ["a/1", "a/2", "b"], tmp_path) == ["a/1", "a/2", "b"]
    http([], encode(document()))
    path, _, _ = fetch_series(config, MANCHESTER, tmp_path)
    assert missing_series(config, ["a/1", "a/2", "b"], tmp_path) == ["b"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PVGISError, match="changed since"):
        missing_series(config, ["a/1"], tmp_path)


# ------------------------------------------------------------------------------ draws


def test_local_days_lie_inside_the_series() -> None:
    """East of UTC the first local day starts before the series, west of it the last ends
    after: neither is drawn."""
    for tz, first, n in (
        ("Europe/London", "2020-01-01", 366),
        ("Europe/Madrid", "2020-01-02", 365),
        ("America/Denver", "2020-01-01", 365),
        ("Asia/Kolkata", "2020-01-02", 365),
    ):
        s = series(tz=tz)
        day, count = local_days(s)
        assert (str(day), count) == (first, n), tz
        last = day + timedelta(days=count - 1)
        midnight = datetime(last.year, last.month, last.day, tzinfo=s.tz) + timedelta(days=1)
        assert midnight.timestamp() <= s.start_s + 3600 * len(s.p_w)


@pytest.mark.parametrize(
    ("tz", "date", "seconds", "utc"),
    [
        # London: GMT in winter, BST (UTC+1) from 29 March 2020 01:00 UTC to 25 October
        ("Europe/London", "2020-03-28", 12 * 3600 + 1800, "2020-03-28T12"),
        ("Europe/London", "2020-03-29", 12 * 3600 + 1800, "2020-03-29T11"),
        ("Europe/London", "2020-07-01", 0, "2020-06-30T23"),
        ("Europe/London", "2020-10-25", 12 * 3600, "2020-10-25T12"),
        ("Europe/London", "2020-03-29", 3600 + 1800, "2020-03-29T01"),  # 01:30 does not exist
        ("Europe/London", "2020-10-25", 3600 + 1800, "2020-10-25T00"),  # 01:30 twice: the first
        # Madrid: CET (UTC+1), CEST (UTC+2) in summer
        ("Europe/Madrid", "2020-01-02", 0, "2020-01-01T23"),
        ("Europe/Madrid", "2020-07-01", 12 * 3600 + 1800, "2020-07-01T10"),
        # Denver: MST (UTC-7), MDT (UTC-6) from 8 March 2020
        ("America/Denver", "2020-03-07", 12 * 3600, "2020-03-07T19"),
        ("America/Denver", "2020-03-08", 12 * 3600, "2020-03-08T18"),
        ("America/Denver", "2020-12-30", 23 * 3600 + 3599, "2020-12-31T06"),
        # Kolkata: UTC+5:30 all year, the record holding the instant
        ("Asia/Kolkata", "2020-06-01", 12 * 3600, "2020-06-01T06"),
        ("Asia/Kolkata", "2020-06-01", 12 * 3600 + 1800, "2020-06-01T07"),
    ],
)
def test_the_load_time_of_day_is_matched_in_local_time(
    tz: str, date: str, seconds: int, utc: str
) -> None:
    s = series(tz=tz)
    first, n_days = local_days(s)
    rng = FixedRng((datetime.fromisoformat(date).date() - first).days)
    i = draw_record(s, rng, seconds)  # type: ignore[arg-type]
    assert rng.bounds == [n_days]
    assert utc_hour(s, i) == datetime.fromisoformat(utc + ":00+00:00")


@pytest.mark.parametrize("tz", ["Europe/London", "America/Denver"])
def test_every_coherent_draw_holds_the_local_hour(tz: str) -> None:
    s = series(document(2019, 2020), tz=tz)
    rng = np.random.default_rng(3)
    for _ in range(500):
        seconds = int(rng.integers(86400))
        i = draw_record(s, rng, seconds)
        local = utc_hour(s, i).astimezone(s.tz)
        if local.hour != seconds // 3600:  # only a wall time skipped in spring: the hour after
            assert local.dst()
            assert local.hour == seconds // 3600 + 1


def test_a_draw_without_load_time_covers_the_whole_series() -> None:
    s = series(tz="Europe/London")
    rng = np.random.default_rng(0)
    drawn = np.array([draw_record(s, rng, None) for _ in range(4000)])
    assert drawn.min() >= 0
    assert drawn.max() < len(s.p_w)
    hours = {utc_hour(s, int(i)).hour for i in drawn}
    assert hours == set(range(24))  # night included
    months = {utc_hour(s, int(i)).month for i in drawn}
    assert months == set(range(1, 13))


# ------------------------------------------------------------------------------ outputs


def units(kwp: list[float], pf: float = 1.0, kva_per_kwp: float = 1.1) -> list[PVUnit]:
    return [
        PVUnit(f"Generator.pv_{i}", f"Load.{i}", k, kva_per_kwp * k * 1e3, pf, 0.44, 0.05)
        for i, k in enumerate(kwp)
    ]


def pvgis_profile(**changes: Any) -> PVProfile:
    return PVProfile.model_validate({"provider": "pvgis", **changes})


def test_available_power_is_kwp_times_the_series_times_availability() -> None:
    s = series(document(p=lambda i: float(i % 1000)), tz="Europe/London")
    fleet = PVOutputs(units([2.0, 4.0]), PV(), pvgis_profile(availability=0.5), s)
    for scenario in range(20):
        draw = fleet.draw(7, 1, 0, scenario, time_s=12 * 3600)
        assert draw.time_utc_s is not None
        i = (draw.time_utc_s - s.start_s) // 3600
        assert np.array_equal(draw.p_avail_w, np.array([2.0, 4.0]) * float(i % 1000) * 0.5)
        assert datetime.fromtimestamp(draw.time_utc_s, s.tz).hour == 12
        assert draw.hour is None
        assert draw.cloudiness is None
        assert np.array_equal(draw.p_w, draw.p_avail_w)  # 1.1 kVA per kWp: no clipping
    times = {fleet.draw(7, 1, 0, scenario, time_s=12 * 3600).time_utc_s for scenario in range(20)}
    assert len(times) == 20  # dates drawn per scenario
    same = fleet.draw(7, 1, 0, 3, time_s=12 * 3600)
    assert same.time_utc_s == fleet.draw(7, 1, 0, 3, time_s=12 * 3600).time_utc_s


def test_a_draw_without_load_time_takes_any_hour() -> None:
    s = series(tz="Europe/London")
    fleet = PVOutputs(units([3.0]), PV(), pvgis_profile(), s)
    hours = set()
    for scenario in range(100):
        draw = fleet.draw(7, 1, 0, scenario, time_s=None)
        assert draw.time_utc_s is not None
        hours.add(datetime.fromtimestamp(draw.time_utc_s, s.tz).hour)
    assert len(hours) > 15


def test_a_drawn_availability_and_an_inverter_limit_apply_as_with_the_clear_sky() -> None:
    s = series(document(p=lambda i: 1200.0), tz="Europe/London")  # a bright, cold hour
    spec = {"distribution": "uniform", "low": 0.5, "high": 1.0}
    fleet = PVOutputs(units([2.0, 5.0], pf=0.9, kva_per_kwp=1.0), PV(), pvgis_profile(), s)
    shaded = PVOutputs(fleet.units, PV(), pvgis_profile(availability=spec), s)
    rated = np.array([2000.0, 5000.0])
    for scenario in range(10):
        draw = fleet.draw(7, 1, 0, scenario, time_s=None)
        assert np.array_equal(draw.p_avail_w, np.array([2400.0, 6000.0]))
        assert np.array_equal(draw.p_w, rated)  # clipped at the rating
        assert not draw.q_var.any()  # no room left for reactive power
        other = shaded.draw(7, 1, 0, scenario, time_s=None)
        assert other.time_utc_s == draw.time_utc_s  # availability has its own stream
        share = other.p_avail_w / draw.p_avail_w
        assert ((share >= 0.5) & (share < 1.0)).all()


def test_the_provider_and_the_series_go_together() -> None:
    with pytest.raises(ValueError, match="provider 'pvgis' with series None"):
        PVOutputs(units([1.0]), PV(), pvgis_profile())
    with pytest.raises(ValueError, match="provider 'synthetic_clear_sky'"):
        PVOutputs(units([1.0]), PV(), PVProfile(), series())


# ------------------------------------------------------------------------------ config

MINIMAL: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["csiro_enwl/n01_*", "ieee13"]},
    "scenarios": {"n_scenarios_per_variant": 10},
    "settings": {"seed": 1},
}
LONDON = {"lat": 53.48, "lon": -2.24, "timezone": "Europe/London"}


def with_pvgis(**pvgis_section: Any) -> dict[str, Any]:
    data = copy.deepcopy(MINIMAL)
    data["scenarios"]["pv_profile"] = {"provider": "pvgis", "pvgis": pvgis_section}
    return data


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


def issues(data: dict[str, Any], catalog: Catalog) -> list[tuple[str, str]]:
    with pytest.raises(ConfigError) as caught:
        validate_config(data, catalog=catalog)
    return caught.value.issues


def test_every_selected_network_needs_a_location(catalog: Catalog) -> None:
    found = issues(with_pvgis(locations={"csiro_enwl/*": LONDON}), catalog)
    assert found == [
        ("scenarios.pv_profile.pvgis.locations", "no location for ieee13 (none is built in)")
    ]
    found = issues(with_pvgis(), catalog)
    assert "no location for csiro_enwl/n01_f1, csiro_enwl/n01_f2" in found[0][1]
    placed = {"csiro_enwl/*": LONDON, "ieee13": {**LONDON, "lat": 40.0}}
    config = validate_config(with_pvgis(locations=placed), catalog=catalog)
    assert config.scenarios.pv_profile.pvgis.matching_keys("csiro_enwl/n01_f1") == ["csiro_enwl/*"]


def test_an_id_wins_over_patterns_and_two_patterns_are_an_error(catalog: Catalog) -> None:
    placed = {"csiro_enwl/*": LONDON, "csiro_enwl/n01_f1": {**LONDON, "lat": 54.0}, "ieee*": LONDON}
    config = validate_config(with_pvgis(locations=placed), catalog=catalog).scenarios.pv_profile
    assert config.pvgis.matching_keys("csiro_enwl/n01_f1") == ["csiro_enwl/n01_f1"]
    assert pvgis.location_of(config.pvgis, "csiro_enwl/n01_f1").lat == 54.0
    assert pvgis.location_of(config.pvgis, "csiro_enwl/n01_f2").lat == 53.48
    twice = {"csiro_enwl/*": LONDON, "csiro_enwl/n01_*": LONDON, "ieee13": LONDON}
    assert issues(with_pvgis(locations=twice), catalog) == [
        (
            "scenarios.pv_profile.pvgis.locations",
            f"{n} is matched by several patterns: csiro_enwl/*, csiro_enwl/n01_*",
        )
        for n in sorted(catalog.networks)
        if n.startswith("csiro_enwl/n01_")
    ]


def test_a_location_must_match_a_catalog_network(catalog: Catalog) -> None:
    placed = {"csiro_enwl/*": LONDON, "ieee13": LONDON, "ieee_13": LONDON}
    assert issues(with_pvgis(locations=placed), catalog) == [
        ("scenarios.pv_profile.pvgis.locations.ieee_13", "'ieee_13' matches no catalog network")
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"startyear": 2004}, "greater than or equal to 2005"),
        ({"endyear": 2024}, "less than or equal to 2023"),
        ({"startyear": 2020, "endyear": 2019}, "startyear 2020 is after endyear 2019"),
        ({"angle": 30}, "angle and aspect are used only with optimalangles: false"),
        ({"optimalangles": False, "angle": 30}, "optimalangles: false needs angle and aspect"),
        ({"optimalangles": False, "angle": 30.5, "aspect": 0}, "valid integer"),
        ({"loss": 100}, "less than 100"),
        ({"pvtechchoice": "mono"}, "'crystSi'"),
        ({"mountingplace": "roof"}, "'free' or 'building'"),
        ({"locations": {"ieee13": {**LONDON, "timezone": "Europe/Manchester"}}}, "unknown IANA"),
        ({"locations": {"ieee13": {**LONDON, "lat": 91}}}, "less than or equal to 90"),
    ],
)
def test_pvgis_settings_are_checked(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PVGIS.model_validate(change)


def test_cloudiness_and_the_pvgis_section_need_their_provider() -> None:
    with pytest.raises(ValueError, match="cloudiness is used only with provider"):
        pvgis_profile(cloudiness=0.5)
    with pytest.raises(ValueError, match="pvgis is used only with provider 'pvgis'"):
        PVProfile.model_validate({"pvgis": {"startyear": 2010}})
    assert pvgis_profile(cloudiness={"distribution": "uniform", "low": 0.0, "high": 1.0})


def test_a_stored_pvgis_config_reads_back(catalog: Catalog) -> None:
    placed = {"csiro_enwl/*": LONDON, "ieee13": LONDON}
    config = validate_config(with_pvgis(locations=placed, loss=10), catalog=catalog)
    assert stored_config(config.model_dump(mode="json")) == config


# ------------------------------------------------------------------------------ CLI


def test_generate_refuses_a_network_without_location_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With and without ``--dry-run``: exit 2, as an invalid config, naming the networks."""
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app
    from gridfm_datakit_distr.networks import cache, catalog
    from tests.test_executor import catalog as test_catalog

    monkeypatch.setattr(catalog, "load_catalog", lambda: test_catalog(tmp_path))
    monkeypatch.setattr(cache, "is_fetched", lambda source, root=None: True)
    config = tmp_path / "c.yaml"
    config.write_text(
        "dataset_name: d\nnetworks: {include: ['t/*']}\nsettings: {seed: 1}\nscenarios:\n"
        "  n_scenarios_per_variant: 1\n  pv_profile:\n    provider: pvgis\n"
    )
    data = tmp_path / "data"
    args = ["generate", str(config), "--cache", str(tmp_path), "--data-dir", str(data)]
    for dry_run in ([], ["--dry-run"]):
        result = CliRunner().invoke(app, [*args, *dry_run])
        assert result.exit_code == 2, result.output
        expected = "- scenarios.pv_profile.pvgis.locations: no location for t/feeder, t/reg"
        assert expected in result.output
        assert "dataset d" not in result.output  # the dry run's report
    assert not data.exists()


def test_generate_refuses_a_missing_or_changed_series_and_names_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, http: Callable[..., FakeHTTP]
) -> None:
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app
    from gridfm_datakit_distr.networks import cache, catalog
    from tests.test_executor import catalog as test_catalog

    monkeypatch.setattr(catalog, "load_catalog", lambda: test_catalog(tmp_path))
    monkeypatch.setattr(cache, "is_fetched", lambda source, root=None: True)
    config = tmp_path / "c.yaml"
    config.write_text(
        "dataset_name: d\nnetworks: {include: ['t/*']}\nsettings: {seed: 1}\nscenarios:\n"
        "  n_scenarios_per_variant: 1\n  pv_profile:\n    provider: pvgis\n    pvgis:\n"
        "      startyear: 2020\n      endyear: 2020\n      locations:\n"
        "        't/*': {lat: 53.48, lon: -2.24, timezone: Europe/London}\n"
    )
    data = tmp_path / "data"
    args = ["generate", str(config), "--cache", str(tmp_path), "--data-dir", str(data)]
    for dry_run in ([], ["--dry-run"]):
        result = CliRunner().invoke(app, [*args, *dry_run])
        assert result.exit_code == 1, result.output
        assert "no PVGIS series for t/feeder, t/reg: run" in result.output
        assert f"fetch --config {config}` first" in result.output
    fake = http([], encode(document()))
    fetched = CliRunner().invoke(app, ["fetch", "--config", str(config), "--cache", str(tmp_path)])
    assert fetched.exit_code == 0, fetched.output
    assert "pvgis 53.48, -2.24: downloaded" in fetched.output
    assert len(fake.calls) == 1  # one location for both networks
    again = CliRunner().invoke(app, ["fetch", "--config", str(config), "--cache", str(tmp_path)])
    assert "already fetched and verified" in again.output
    path = series_path(fake.calls[0], tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")
    result = CliRunner().invoke(app, [*args, "--dry-run"])
    assert result.exit_code == 1, result.output
    assert "changed since it was downloaded" in result.output
    assert not data.exists()


def test_fetch_needs_ids_or_a_config() -> None:
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app

    result = CliRunner().invoke(app, ["fetch"])
    assert result.exit_code == 2
    assert "give source or network ids, or --config" in result.output


# ------------------------------------------------------------------------------ dataset


def test_a_pvgis_dataset_records_its_series(tmp_path: Path, http: Callable[..., FakeHTTP]) -> None:
    """Two networks at one location: each scenario's PV hour is stored, the available power
    is kWp times P of that hour, and the manifest, the passports and the card name the series."""
    import pyarrow.parquet as pq

    from gridfm_datakit_distr.config.schema import GenerationConfig
    from gridfm_datakit_distr.io.manifest import Manifest
    from gridfm_datakit_distr.solve.executor import GenerationError, generate
    from tests.test_executor import CONFIG
    from tests.test_executor import catalog as test_catalog

    data = copy.deepcopy(CONFIG)
    data["der"] = {"pv": {"penetration": 1.0}}
    data["settings"]["float_precision"] = "full"
    data["scenarios"]["pv_profile"] = {
        "provider": "pvgis",
        "pvgis": {"startyear": 2020, "endyear": 2020, "locations": {"t/*": LONDON}},
    }
    config = GenerationConfig.model_validate(data)
    http([], encode(document(p=lambda i: float(i % 997))))
    path, s, _ = fetch_series(config.scenarios.pv_profile.pvgis, MANCHESTER, tmp_path)
    out = tmp_path / "out"
    generate(config, out, catalog=test_catalog(tmp_path), root=tmp_path)

    meta = pq.read_table(out / "scenarios" / "scenario_meta").to_pylist()
    assert all(m["pv_time_utc_s"] is not None for m in meta)
    assert len({m["pv_time_utc_s"] for m in meta}) == len(meta)  # any hour: synthetic loads
    variants = {
        v["variant_id"]: json.loads(v["perturbation"])["pv_kwp"]
        for v in pq.read_table(out / "variants" / "variants").to_pylist()
    }
    der = pq.read_table(out / "scenarios" / "der").to_pylist()
    for m in meta:
        p = float(((m["pv_time_utc_s"] - s.start_s) // 3600) % 997)
        kwp = [kw for _, kw in sorted(variants[m["variant_id"]].items())]
        rows = [r for r in der if r["scenario_id"] == m["scenario_id"] and r["der_type"] == "pv"]
        assert [r["p_avail_w"] for r in rows] == [k * p for k in kwp]

    manifest = Manifest.load(out)
    record = manifest.networks[0]["pvgis"]
    assert record == manifest.networks[1]["pvgis"]
    assert record["sha256"] == s.sha256
    assert record["request_url"] == request_url(config.scenarios.pv_profile.pvgis, MANCHESTER)
    assert (record["radiation_db"], record["slope_deg"]) == ("PVGIS-SARAH3", 40.0)
    assert record["attribution"] == "PVGIS © European Communities, 2001-2026"
    passport = json.loads((out / "catalog" / "passports" / "t" / "reg.json").read_text())
    assert passport["pv_series"] == record
    card = (out / "README.md").read_text()
    assert "PVGIS © European Communities, 2001-2026" in card
    assert "53.48, -2.24 (Europe/London), 2020-2020, PVGIS-SARAH3 and ERA5" in card
    assert "synthetic clear-sky" not in card

    path.write_bytes(encode(document(p=lambda i: 1.0)))  # another series, recorded as fetched
    path.with_name(path.name + ".sha256").write_text(
        f"{pvgis.hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
    )
    pvgis._cached.cache_clear()
    with pytest.raises(GenerationError, match="not the one this dataset started with"):
        generate(config, out, catalog=test_catalog(tmp_path), root=tmp_path)
