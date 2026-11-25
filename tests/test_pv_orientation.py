"""Per-unit PV orientation (``scenarios.pv_profile.pvgis.orientation``): config checks, the class
drawn per unit and variant, each unit reading its class's PVGIS series, a dataset.

Series are made-up numbers in PVGIS's JSON layout (``tests.test_pvgis``); no network access.
"""

from __future__ import annotations

import copy
import io
import json
import urllib.parse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import PV, PVGIS, PVOrientation, PVProfile, Variants
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.scenarios import pvgis
from gridfm_datakit_distr.scenarios.der import PVOutputs, pv_units
from gridfm_datakit_distr.scenarios.seeds import VariantStream
from gridfm_datakit_distr.variants import perturbation
from gridfm_datakit_distr.variants.perturbation import sample_perturbation
from gridfm_datakit_distr.variants.targets import VariantTargets, collect_targets
from tests.circuits import VARIANT_FEEDER, write_circuit
from tests.test_pvgis import LONDON, document, encode, pvgis_profile, series, units

ORIENTED: dict[str, Any] = {"optimalangles": False, "orientation": {"enabled": True}}
SEED = 5


# ------------------------------------------------------------------------------ config


def test_the_default_classes_are_25_pairs_and_21_series() -> None:
    o = PVOrientation()
    assert not o.enabled
    assert (o.tilt_deg, o.azimuth_deg) == ([0, 15, 30, 45, 60], [-90, -45, 0, 45, 90])
    pairs = o.pairs()
    assert len(pairs) == 25
    assert pairs.count((0, 0)) == 5  # every tilt-0 pair reads the horizontal series
    assert len(set(pairs)) == 21
    settings = pvgis.class_settings(PVGIS.model_validate(ORIENTED))
    assert [(s.angle, s.aspect) for s in settings] == list(dict.fromkeys(pairs))
    assert all(not s.optimalangles and not s.orientation.enabled for s in settings)
    assert pvgis.class_settings(PVGIS()) == [PVGIS()]


def test_orientation_needs_pvgis_and_says_why() -> None:
    with pytest.raises(ValueError, match="orientation needs provider 'pvgis': the synthetic "):
        PVProfile.model_validate({"pvgis": ORIENTED})
    with pytest.raises(ValueError, match="latitude 0, where a tilt toward the equator only"):
        PVProfile.model_validate({"provider": "synthetic_clear_sky", "pvgis": ORIENTED})
    assert pvgis_profile(pvgis=ORIENTED).pvgis.orientation.enabled


@pytest.mark.parametrize(
    ("section", "message"),
    [
        # optimalangles defaults to true: the first error, whatever the provider
        ({"orientation": {"enabled": True}}, "incompatible with optimalangles: true"),
        ({**ORIENTED, "angle": 30, "aspect": 0}, "incompatible with optimalangles: true and with"),
        ({**ORIENTED, "aspect": 0}, "leave angle and aspect unset"),
        ({"optimalangles": False}, "optimalangles: false needs angle and aspect, or orientation"),
        ({**ORIENTED, "orientation": {"enabled": True, "tilt_deg": [91]}}, "less than or equal"),
        ({**ORIENTED, "orientation": {"tilt_deg": [-1]}}, "greater than or equal to 0"),
        ({**ORIENTED, "orientation": {"tilt_deg": [30.5]}}, "valid integer"),
        ({**ORIENTED, "orientation": {"azimuth_deg": [181]}}, "less than or equal to 180"),
        ({**ORIENTED, "orientation": {"azimuth_deg": []}}, "at least 1 item"),
        ({**ORIENTED, "orientation": {"tilt_deg": [30, 30]}}, "tilt_deg has a duplicate"),
    ],
)
def test_orientation_settings_are_checked(section: dict[str, Any], message: str) -> None:
    for provider in ("pvgis", "synthetic_clear_sky"):
        with pytest.raises(ValueError, match=message):
            PVProfile.model_validate({"provider": provider, "pvgis": section})


# ------------------------------------------------------------------------------ draw


@pytest.fixture(scope="module")
def targets(tmp_path_factory: pytest.TempPathFactory) -> VariantTargets:
    root = tmp_path_factory.mktemp("orientation")
    write_circuit(root / "src", VARIANT_FEEDER)
    network = Network(id="t/orientation", source="src", master="master.dss")
    return collect_targets(prepare_network(network, root=root))


def draw(targets: VariantTargets, k: int, rho: float = 1.0, **kwargs: Any) -> Any:
    der = PV.model_validate({"penetration": rho})
    return sample_perturbation(Variants(), targets, SEED, 0, k, der=der, **kwargs)


def test_the_class_is_drawn_per_unit_and_variant_from_its_own_stream(
    targets: VariantTargets, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = PVOrientation(enabled=True)
    first = draw(targets, 0, orientation=spec)
    assert set(first.pv_orientation) == set(first.pv_kwp)
    assert first == draw(targets, 0, orientation=spec)  # deterministic
    plain = draw(targets, 0)
    assert replace(first, pv_orientation={}) == plain  # no other draw moves
    assert draw(targets, 0, orientation=PVOrientation()) == plain  # disabled: nothing drawn
    assert "pv_orientation" not in json.loads(plain.to_json())
    recorded = json.loads(first.to_json())["pv_orientation"]
    assert recorded == {e: list(c) for e, c in first.pv_orientation.items()}
    half = draw(targets, 0, rho=0.5, orientation=spec)  # a unit keeps its class
    assert half.pv_orientation == {e: first.pv_orientation[e] for e in half.pv_orientation}
    assert [u.orientation for u in pv_units(first, PV())] == [
        first.pv_orientation[e] for e in sorted(first.pv_kwp)
    ]

    streams: list[VariantStream] = []
    real = perturbation.variant_rng

    def recording(seed: int, net: int, k: int, stream: VariantStream, attempt: int = 0) -> Any:
        streams.append(stream)
        return real(seed, net, k, stream, attempt)

    monkeypatch.setattr(perturbation, "variant_rng", recording)
    draw(targets, 0)
    assert VariantStream.PV_ORIENTATION not in streams
    draw(targets, 0, orientation=spec)
    assert VariantStream.PV_ORIENTATION in streams


def test_the_classes_are_uniform_over_the_pairs(targets: VariantTargets) -> None:
    spec = PVOrientation(enabled=True)
    counts = Counter(
        c for k in range(300) for c in draw(targets, k, orientation=spec).pv_orientation.values()
    )
    n = sum(counts.values())
    assert set(counts) == set(spec.pairs())
    assert abs(counts[(0, 0)] / n - 5 / 25) < 0.03
    assert max(abs(counts[c] / n - 1 / 25) for c in counts if c != (0, 0)) < 0.02


# ------------------------------------------------------------------------------ outputs


def hump(peak: int) -> Any:
    """``P`` of a series peaking at ``peak`` h UTC every day."""
    return lambda i: float(max(0, 1000 - 150 * abs(i % 24 - peak)))


def test_units_with_different_orientations_peak_at_different_hours() -> None:
    east, west = series(document(p=hump(9))), series(document(p=hump(15)))
    classes = {(30, -90): east, (30, 90): west}
    oriented = [replace(u, orientation=c) for u, c in zip(units([2.0, 3.0]), classes, strict=True)]
    fleet = PVOutputs(oriented, PV(), pvgis_profile(pvgis=ORIENTED), classes)
    p = np.array([fleet.draw(7, 1, 0, 0, time_s=h * 3600).p_avail_w for h in range(24)])
    assert p.argmax(axis=0).tolist() == [9, 15]
    for s in range(10):  # one hour for every unit
        d = fleet.draw(7, 1, 0, s, time_s=None)
        assert d.time_utc_s is not None
        i = (d.time_utc_s - east.start_s) // 3600
        assert d.p_avail_w.tolist() == [2.0 * east.p_w[i], 3.0 * west.p_w[i]]


def test_units_and_series_must_agree_on_orientation() -> None:
    s = series()
    with pytest.raises(ValueError, match="need a series per class"):
        PVOutputs([replace(units([1.0])[0], orientation=(30, 0))], PV(), pvgis_profile(), s)
    with pytest.raises(ValueError, match="needs units with a class"):
        PVOutputs(units([1.0]), PV(), pvgis_profile(), {(30, 0): s})


# ------------------------------------------------------------------------------ dataset


def class_power(angle: int, aspect: int) -> Any:
    """``P`` of the made-up series of a class: a different number per class and hour."""
    return lambda i: float((i + 7 * angle + aspect + 180) % 997)


def answer(request: Any) -> io.BytesIO:
    """PVGIS answering each request with the angles it was asked for."""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
    angle, aspect = int(query["angle"][0]), int(query["aspect"][0])
    doc = document(p=class_power(angle, aspect))
    doc["inputs"]["mounting_system"]["fixed"] |= {
        "slope": {"value": angle, "optimal": False},
        "azimuth": {"value": aspect, "optimal": False},
    }
    return io.BytesIO(encode(doc))


def test_an_oriented_dataset_reads_each_units_class_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fetch --config`` downloads 21 series per location; each unit's available power is its
    kWp times its class's ``P`` in the scenario's hour; manifest, passports and card list the
    series; a resumed dataset refuses a changed one."""
    import pyarrow.parquet as pq
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app
    from gridfm_datakit_distr.config.load import stored_config
    from gridfm_datakit_distr.config.schema import GenerationConfig
    from gridfm_datakit_distr.io.manifest import Manifest
    from gridfm_datakit_distr.networks import catalog
    from gridfm_datakit_distr.solve.executor import GenerationError, generate
    from tests.test_executor import CONFIG
    from tests.test_executor import catalog as test_catalog

    data = copy.deepcopy(CONFIG)
    data["der"] = {"pv": {"penetration": 1.0}}
    data["settings"]["float_precision"] = "full"
    section = {**ORIENTED, "startyear": 2020, "endyear": 2020, "locations": {"t/*": LONDON}}
    data["scenarios"]["pv_profile"] = {"provider": "pvgis", "pvgis": section}
    config_path = tmp_path / "c.yaml"
    config_path.write_text(json.dumps(data))  # JSON is YAML
    asked: list[str] = []
    monkeypatch.setattr(
        pvgis.urllib.request, "urlopen", lambda r, timeout: asked.append(r.full_url) or answer(r)
    )
    monkeypatch.setattr(catalog, "load_catalog", lambda: test_catalog(tmp_path))
    fetched = CliRunner().invoke(
        app, ["fetch", "--config", str(config_path), "--cache", str(tmp_path)]
    )
    assert fetched.exit_code == 0, fetched.output
    assert len(asked) == fetched.output.count(": downloaded") == 21
    assert "slope 15, azimuth -90" in fetched.output

    config = GenerationConfig.model_validate(data)
    assert stored_config(config.model_dump(mode="json")) == config
    pv_config = config.scenarios.pv_profile.pvgis
    assert pvgis.missing_series(pv_config, ["t/feeder", "t/reg"], tmp_path) == []
    one = pvgis.series_path(asked[3], tmp_path)
    one.rename(one.with_suffix(".away"))  # one class missing: every network at the location
    assert pvgis.missing_series(pv_config, ["t/feeder", "t/reg"], tmp_path) == ["t/feeder", "t/reg"]
    one.with_suffix(".away").rename(one)
    out = tmp_path / "out"
    generate(config, out, catalog=test_catalog(tmp_path), root=tmp_path)

    meta = pq.read_table(out / "scenarios" / "scenario_meta").to_pylist()
    start = series().start_s
    variants = {
        v["variant_id"]: json.loads(v["perturbation"])
        for v in pq.read_table(out / "variants" / "variants").to_pylist()
    }
    der = pq.read_table(out / "scenarios" / "der").to_pylist()
    seen = set()
    for m in meta:
        i = (m["pv_time_utc_s"] - start) // 3600
        drawn = variants[m["variant_id"]]
        kwp, classes = drawn["pv_kwp"], drawn["pv_orientation"]
        rows = [r for r in der if r["scenario_id"] == m["scenario_id"] and r["der_type"] == "pv"]
        expected = [kwp[e] * class_power(*classes[e])(i) for e in sorted(kwp)]
        assert [r["p_avail_w"] for r in rows] == expected
        seen |= {tuple(c) for c in classes.values()}
    assert len(seen) > 1  # units of the dataset read different series

    manifest = Manifest.load(out)
    records = manifest.networks[0]["pvgis"]
    assert records == manifest.networks[1]["pvgis"]
    assert [(r["slope_deg"], r["azimuth_deg"]) for r in records] == list(
        dict.fromkeys(pv_config.orientation.pairs())
    )
    assert len({r["sha256"] for r in records}) == 21
    passport = json.loads((out / "catalog" / "passports" / "t" / "reg.json").read_text())
    assert passport["pv_series"] == records
    card = (out / "README.md").read_text()
    assert "PVGIS-SARAH3 and ERA5, slope 60°, azimuth 90° for `t/feeder`, `t/reg`" in card
    assert "PVGIS © European Communities, 2001-2026" in card

    path = pvgis.series_path(asked[-1], tmp_path)
    path.write_bytes(encode(document(p=lambda i: 1.0)))  # another series, recorded as fetched
    path.with_name(path.name + ".sha256").write_text(
        f"{pvgis.hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
    )
    pvgis._cached.cache_clear()
    with pytest.raises(GenerationError, match="not the one this dataset started with"):
        generate(config, out, catalog=test_catalog(tmp_path), root=tmp_path)
