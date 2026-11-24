"""PV output series of PVGIS 5.3 (EU Joint Research Centre), ``scenarios.pv_profile.provider:
pvgis``.

One ``seriescalc`` request per location and PV-parameter set (``pvcalculation=1``,
``peakpower=1``, every configured year, JSON) gives the hourly power ``P`` (W) of a 1 kWp
system, with PVGIS's temperature model and system loss. ``fetch --config`` downloads it into
``<cache>/pvgis/<key>.json`` (``key``: the first 16 hex digits of the SHA-256 of the request URL)
and records the file's SHA-256 next to it (``<key>.json.sha256``, ``sha256sum`` format). Every
later use checks the file against it: a changed file is an error, never a silent download.

Timestamps are taken as UTC (``seriescalc`` does not say; the sun height agrees). A record stands
for the hour that starts at its ``HH:00``: SARAH3 values are stamped about HH:10 and taken by
PVGIS as the hour's average, ERA5 values are hourly averages. A series must hold every hour from
1 January 00:00 of its first year to 31 December 23:00 of its last, in order, with a finite,
non-negative ``P``; anything else is refused.

Per scenario (:func:`draw_record`, stream ``PV_TIME``): with a load time of day, a date drawn
uniformly among the local days the series covers entirely and the record holding that local
time on that date (``zoneinfo``, so DST applies; a wall time that does not exist or occurs twice
takes the UTC offset in force before the change, ``fold=0``); without one, any record uniformly.
"""

from __future__ import annotations

import functools
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import PVGIS, PVGISLocation, PVProfile
from gridfm_datakit_distr.networks.cache import cache_root
from gridfm_datakit_distr.networks.fetch import USER_AGENT

API = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
HOMEPAGE = (
    "https://joint-research-centre.ec.europa.eu/"
    "photovoltaic-geographical-information-system-pvgis_en"
)
ATTRIBUTION = "PVGIS © European Communities, 2001-2026"
CACHE_DIR = "pvgis"
RETRY_STATUS = {429: "too many requests", 529: "overloaded"}
"""Statuses retried (PVGIS allows 30 requests per second); any other is an error."""
ATTEMPTS = 4
TIMEOUT_S = 300.0
"""A series of 19 years is about 16 MB."""
_MOUNTING = {"free": "free-standing", "building": "building-integrated"}
"""``mountingplace`` as the response reports it."""


class PVGISError(RuntimeError):
    """A PVGIS series that cannot be downloaded, read or trusted."""


@dataclass(frozen=True, slots=True, eq=False)
class PVSeries:
    """A validated PVGIS series."""

    start_s: int
    """Start of the first record's hour (Unix time, s)."""
    p_w: NDArray[np.float64]
    """Power of a 1 kWp system in each hour (W), record ``i`` at ``start_s + 3600 i``."""
    radiation_db: str
    meteo_db: str
    slope_deg: float
    azimuth_deg: float
    sha256: str
    tz: tzinfo = timezone.utc
    """The location's time zone: local time of day of the draws."""


# ------------------------------------------------------------------------------ locations


def location_of(pvgis: PVGIS, network_id: str) -> PVGISLocation:
    """The location of a network (``PVGIS.matching_keys``: exactly one key)."""
    keys = pvgis.matching_keys(network_id)
    if len(keys) != 1:
        found = "no location" if not keys else f"several locations ({', '.join(keys)})"
        raise PVGISError(f"{network_id}: {found} in scenarios.pv_profile.pvgis.locations")
    return pvgis.locations[keys[0]]


def request_url(pvgis: PVGIS, location: PVGISLocation) -> str:
    """The ``seriescalc`` request of a location, parameters in a fixed order."""
    if pvgis.optimalangles:
        angles: dict[str, Any] = {"optimalangles": 1}
    else:
        angles = {"angle": pvgis.angle, "aspect": pvgis.aspect}
    params = {
        "lat": repr(float(location.lat)),
        "lon": repr(float(location.lon)),
        "startyear": pvgis.startyear,
        "endyear": pvgis.endyear,
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": repr(float(pvgis.loss)),
        "pvtechchoice": pvgis.pvtechchoice,
        "mountingplace": pvgis.mountingplace,
        **angles,
        "outputformat": "json",
    }
    return f"{API}?{urllib.parse.urlencode(params)}"


def series_path(url: str, root: Path | None = None) -> Path:
    """Where the series of a request is cached."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return (root if root is not None else cache_root()) / CACHE_DIR / f"{key}.json"


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


# ------------------------------------------------------------------------------ download


def fetch_series(
    pvgis: PVGIS, location: PVGISLocation, root: Path | None = None, *, force: bool = False
) -> tuple[Path, PVSeries, bool]:
    """The cached series of a location, downloaded if absent (``force``: downloaded again):
    its path, the series and whether it was downloaded. A cached copy is checked first and
    never replaced without ``force``."""
    url = request_url(pvgis, location)
    path = series_path(url, root)
    if path.is_file() and not force:
        return path, read_series(path), False
    data = _get(url)
    doc = _document(data, url)
    series = _series(doc, hashlib.sha256(data).hexdigest(), url)
    _check_echo(doc, pvgis, location, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_bytes(data)
    _checksum_path(path).write_text(f"{series.sha256}  {path.name}\n", encoding="utf-8")
    staged.replace(path)
    return path, series, True


def _get(url: str) -> bytes:
    """GET, retried on 429 and 529 after 1, 2, 4 s."""
    delay = 1.0
    for attempt in range(1, ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as exc:
            if exc.code in RETRY_STATUS and attempt < ATTEMPTS:
                time.sleep(delay)
                delay *= 2
                continue
            if exc.code in RETRY_STATUS:
                raise PVGISError(
                    f"PVGIS answered HTTP {exc.code} ({RETRY_STATUS[exc.code]}) {attempt} "
                    f"times; try again later: {url}"
                ) from exc
            message = _error_message(exc)
            raise PVGISError(
                f"PVGIS refused the request (HTTP {exc.code}{': ' + message if message else ''})"
                f": {url}"
            ) from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as exc:
            raise PVGISError(f"GET {url}: {exc}") from exc
    raise AssertionError("unreachable")


def _error_message(exc: urllib.error.HTTPError) -> str | None:
    """PVGIS's explanation of an error (``{"message": ..., "status": ...}``), if any."""
    try:
        message = json.loads(exc.read()).get("message")
    except (OSError, ValueError, AttributeError):
        return None
    return str(message) if message else None


# ------------------------------------------------------------------------------ reading


@functools.lru_cache(maxsize=8)
def _cached(path: Path) -> PVSeries:
    return read_series(path)


def read_series(path: Path) -> PVSeries:
    """A cached series, checked against the SHA-256 recorded when it was downloaded."""
    listed = _checksum_path(path)
    again = "download it again with `gridfm_datakit_distr fetch --config <config> --force`"
    if not path.is_file():
        raise PVGISError(f"{path}: no such PVGIS series")
    if not listed.is_file():
        raise PVGISError(f"{path}: no SHA-256 recorded at download ({listed.name}); {again}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    words = listed.read_text(encoding="utf-8").split()
    recorded = words[0] if words else ""
    if digest != recorded:
        raise PVGISError(
            f"{path}: changed since it was downloaded (SHA-256 {digest}, recorded "
            f"{recorded or 'nothing'}); {again}"
        )
    return _series(_document(data, str(path)), digest, str(path))


def series_for(profile: PVProfile, network_id: str, root: Path | None = None) -> PVSeries | None:
    """The series of a network in its time zone, read once per process; None with another
    provider."""
    if profile.provider != "pvgis":
        return None
    location = location_of(profile.pvgis, network_id)
    series = _cached(series_path(request_url(profile.pvgis, location), root))
    return replace(series, tz=ZoneInfo(location.timezone))


def missing_series(pvgis: PVGIS, network_ids: Sequence[str], root: Path | None = None) -> list[str]:
    """The networks whose series is not in the cache; the cached ones are checked
    (:class:`PVGISError` if one changed)."""
    missing, checked = [], set()
    for network_id in network_ids:
        path = series_path(request_url(pvgis, location_of(pvgis, network_id)), root)
        if not path.is_file():
            missing.append(network_id)
        elif path not in checked:
            read_series(path)
            checked.add(path)
    return missing


def series_record(pvgis: PVGIS, network_id: str, root: Path | None = None) -> dict[str, Any]:
    """A network's series as ``manifest.json`` (``networks[].pvgis``) and its passport
    (``pv_series``) record it: request, databases and angles PVGIS used, checksum, attribution."""
    location = location_of(pvgis, network_id)
    url = request_url(pvgis, location)
    series = _cached(series_path(url, root))
    return {
        "source": "PVGIS 5.3 seriescalc, EU Joint Research Centre",
        "attribution": ATTRIBUTION,
        "source_url": HOMEPAGE,
        "request_url": url,
        "latitude": location.lat,
        "longitude": location.lon,
        "timezone": location.timezone,
        "years": [pvgis.startyear, pvgis.endyear],
        "radiation_db": series.radiation_db,
        "meteo_db": series.meteo_db,
        "slope_deg": series.slope_deg,
        "azimuth_deg": series.azimuth_deg,
        "sha256": series.sha256,
    }


def _document(data: bytes, origin: str) -> dict[str, Any]:
    try:
        doc = json.loads(data)
    except ValueError as exc:
        raise PVGISError(f"{origin}: not a JSON document ({exc})") from None
    if not isinstance(doc, dict):
        raise PVGISError(f"{origin}: not a PVGIS response (a JSON {type(doc).__name__})")
    return doc


def _series(doc: Mapping[str, Any], sha256: str, origin: str) -> PVSeries:
    """The series of a response; :class:`PVGISError` unless every hour is there, in order."""
    try:
        meteo = doc["inputs"]["meteo_data"]
        fixed = doc["inputs"]["mounting_system"]["fixed"]
        first, last = int(meteo["year_min"]), int(meteo["year_max"])
        radiation_db, meteo_db = str(meteo["radiation_db"]), str(meteo["meteo_db"])
        slope, azimuth = float(fixed["slope"]["value"]), float(fixed["azimuth"]["value"])
        records = doc["outputs"]["hourly"]
        stamps = [r["time"] for r in records]
        p = np.array([r["P"] for r in records], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise PVGISError(
            f"{origin}: not a PVGIS seriescalc response with PV power ({type(exc).__name__}: {exc})"
        ) from None
    hours = _hours(stamps, origin)
    start = np.datetime64(f"{first:04d}-01-01T00", "h")
    expected = np.arange(start, np.datetime64(f"{last + 1:04d}-01-01T00", "h"))
    if len(hours) != len(expected) or not np.array_equal(hours, expected):
        n = min(len(hours), len(expected))
        differ = np.flatnonzero(hours[:n] != expected[:n])
        i = int(differ[0]) if differ.size else n
        found = str(hours[i]) if i < len(hours) else "nothing"
        wanted = str(expected[i]) if i < len(expected) else "nothing"
        raise PVGISError(
            f"{origin}: {len(hours)} hourly records where {first}-{last} has {len(expected)}; "
            f"record {i} is {found}, expected {wanted} (a missing, repeated or extra hour)"
        )
    bad = np.flatnonzero(~np.isfinite(p) | (p < 0))
    if bad.size:
        i = int(bad[0])
        raise PVGISError(f"{origin}: record {i} ({stamps[i]}) has P = {p[i]}, not a power")
    start_s = int(start.astype("datetime64[s]").astype(np.int64))
    return PVSeries(start_s, p, radiation_db, meteo_db, slope, azimuth, sha256)


def _hours(stamps: Sequence[object], origin: str) -> NDArray[np.datetime64]:
    """The hour each record stands for: its ``YYYYMMDD:HHMM`` stamp floored to ``HH:00``."""
    iso = []
    for i, t in enumerate(stamps):
        ok = isinstance(t, str) and len(t) == 13 and t[8] == ":" and (t[:8] + t[9:]).isdigit()
        if not ok or int(str(t)[11:]) > 59:
            raise PVGISError(f"{origin}: record {i} has time {t!r}, not YYYYMMDD:HHMM")
        iso.append(f"{str(t)[:4]}-{str(t)[4:6]}-{str(t)[6:8]}T{str(t)[9:11]}")
    try:
        return np.array(iso, dtype="datetime64[h]")
    except ValueError as exc:
        raise PVGISError(f"{origin}: a record has an invalid time ({exc})") from None


def _check_echo(doc: Mapping[str, Any], pvgis: PVGIS, location: PVGISLocation, origin: str) -> None:
    """The response must report the request: PVGIS ignores what it does not parse (of a
    repeated parameter it keeps the first) and rounds angles to whole degrees."""
    try:
        inputs = doc["inputs"]
        fixed = inputs["mounting_system"]["fixed"]
        got = {
            "lat": inputs["location"]["latitude"],
            "lon": inputs["location"]["longitude"],
            "startyear": inputs["meteo_data"]["year_min"],
            "endyear": inputs["meteo_data"]["year_max"],
            "peakpower": inputs["pv_module"]["peak_power"],
            "loss": inputs["pv_module"]["system_loss"],
            "mountingplace": fixed["type"],
            "optimalangles": fixed["slope"]["optimal"] and fixed["azimuth"]["optimal"],
        }
        if not pvgis.optimalangles:
            got |= {"angle": fixed["slope"]["value"], "aspect": fixed["azimuth"]["value"]}
    except (KeyError, TypeError) as exc:
        raise PVGISError(f"{origin}: the response does not report its inputs ({exc})") from None
    wanted: dict[str, Any] = {
        "lat": location.lat,
        "lon": location.lon,
        "startyear": pvgis.startyear,
        "endyear": pvgis.endyear,
        "peakpower": 1.0,
        "loss": pvgis.loss,
        "mountingplace": _MOUNTING[pvgis.mountingplace],
        "optimalangles": pvgis.optimalangles,
    }
    if not pvgis.optimalangles:
        wanted |= {"angle": pvgis.angle, "aspect": pvgis.aspect}
    differ = [f"{k} {got[k]!r} (asked {v!r})" for k, v in wanted.items() if got[k] != v]
    if differ:
        raise PVGISError(f"{origin}: the response reports other inputs: {', '.join(differ)}")


# ------------------------------------------------------------------------------ draws


_DAY = timedelta(days=1)


def local_days(series: PVSeries) -> tuple[date, int]:
    """The first local date the series covers entirely and the number of such dates."""
    end_s = series.start_s + 3600 * len(series.p_w)
    start = datetime.fromtimestamp(series.start_s, timezone.utc).astimezone(series.tz)
    end = datetime.fromtimestamp(end_s, timezone.utc).astimezone(series.tz)
    first = start.date() if start.time() == datetime.min.time() else start.date() + _DAY
    return first, (end.date() - _DAY - first).days + 1


def draw_record(series: PVSeries, rng: np.random.Generator, time_of_day_s: int | None) -> int:
    """Index of the record of a scenario. ``time_of_day_s``: the load profile's local time of
    day (s), drawn on a uniform date among :func:`local_days`; None: any record, uniformly."""
    n = len(series.p_w)
    if time_of_day_s is None:
        return int(rng.integers(n))
    first, n_days = local_days(series)
    day = first + timedelta(days=int(rng.integers(n_days)))
    wall = datetime(day.year, day.month, day.day, tzinfo=series.tz) + timedelta(
        seconds=time_of_day_s
    )  # wall-clock arithmetic: the zone's offset of that date and time, fold=0
    i = (wall.astimezone(timezone.utc).timestamp() - series.start_s) // 3600
    if not 0 <= i < n:
        raise PVGISError(f"{wall.isoformat()} is outside the PVGIS series")
    return int(i)
