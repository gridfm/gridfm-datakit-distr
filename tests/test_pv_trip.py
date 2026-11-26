"""Over- and under-voltage trip of the PV units (``der.pv.trip``)."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.load import ConfigError, validate_config
from gridfm_datakit_distr.config.schema import (
    PV,
    TRIP_NATIONAL,
    TRIP_OV_UNION,
    TRIP_UV_UNION,
    GenerationConfig,
    PVTrip,
    Variants,
)
from gridfm_datakit_distr.config.value_spec import Fixed
from gridfm_datakit_distr.engine.context import SolveStatus
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.networks.catalog import ArchivesSource, Catalog, Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.scenarios.der import pair_base_v, protection_pairs
from gridfm_datakit_distr.solve.executor import GenerationError, generate, jobs_of
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.perturbation import sample_perturbation
from gridfm_datakit_distr.variants.targets import collect_targets
from tests.circuits import (
    DG_FEEDER,
    OPF_DEVICES,
    OPF_LOADS,
    SPLIT_PHASE,
    VARIANT_FEEDER,
    write_circuit,
)

SEED = 11
TRIP = {"enabled": True, "ov_pu": 1.10, "uv_pu": 0.70}
BASE: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["t/lv"]},
    "variants": {"n_variants": 1, "phase_reallocation": {"enabled": False}},
    "der": {"pv": {"penetration": 1.0, "kwp": 10.0, "trip": TRIP}},
    "scenarios": {"n_scenarios_per_variant": 8, "source_voltage_pu": 1.0},
    "controls": {"mode": "as_is"},
    "settings": {"seed": SEED, "residual_sample_rate": 1.0},
}


def config(**changes: Any) -> GenerationConfig:
    """``BASE`` with ``changes`` merged one level deep under each section."""
    data = copy.deepcopy(BASE)
    for section, value in changes.items():
        data[section] = {**data.get(section, {}), **value}
    return GenerationConfig.model_validate(data)


def runner(root: Path, cfg: GenerationConfig, circuit: dict[str, str] = OPF_LOADS) -> VariantRunner:
    write_circuit(root / "src" / "lv", circuit)
    network = Network(id="t/lv", source="src", master="lv/master.dss")
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options, root))
    trip = cfg.der.pv.trip.settings(None) if cfg.der.pv.trip.enabled else None
    result = generate_variant(
        network, targets, cfg.variants, SEED, 0, options=options, root=root, der=cfg.der.pv,
        pv_trip=trip,
    )  # fmt: skip
    assert result.status == "ok"
    return VariantRunner(network, result, cfg, 0, root=root)


def catalog(root: Path, country: str | None) -> Catalog:
    write_circuit(root / "src" / "lv", OPF_LOADS)
    write_circuit(root / "src" / "dg", DG_FEEDER)
    source = ArchivesSource(
        kind="archives", id="src", title="test", homepage="-", license="CC0-1.0",
        citation="-", archives={}, country=country,
    )  # fmt: skip
    networks = {
        "t/lv": Network(id="t/lv", source="src", master="lv/master.dss"),
        "t/dg": Network(id="t/dg", source="src", master="dg/master.dss"),
    }
    return Catalog(sources={"src": source}, networks=networks)


# ------------------------------------------------------------------- what a unit reads


def test_a_unit_reads_its_own_terminal_voltages(tmp_path: Path) -> None:
    """Wye: each phase to its neutral (four-wire) or to ground; a wye unit on several phases also
    phase to phase; delta: phase to phase. Bases: the nominal line-to-neutral voltage, √3 times
    it between phases, 2 times it across a split-phase service."""
    write_circuit(tmp_path / "src" / "d", OPF_DEVICES)
    model = prepare_network(
        Network(id="d", source="src", master="d/master.dss"), root=tmp_path
    ).model
    by_name = {e.name.lower(): e for e in model.elements}

    def labels(name: str) -> list[tuple[str, str, float]]:
        pairs = protection_pairs(model, by_name[name])
        conductor = [model.nodes[p].conductor for p, _ in pairs]
        other = ["g" if q == GROUND else model.nodes[q].conductor for _, q in pairs]
        ratio = [pair_base_v(model, p, q) / model.nodes[p].nominal_v_ln for p, q in pairs]
        return [(a, b, round(r, 12)) for a, b, r in zip(conductor, other, ratio, strict=True)]

    r3 = round(math.sqrt(3.0), 12)
    assert labels("generator.pv_house") == [("c", "n", 1.0)]
    assert labels("generator.pv_wye3") == [
        ("a", "n", 1.0), ("b", "n", 1.0), ("c", "n", 1.0),
        ("a", "b", r3), ("b", "c", r3), ("c", "a", r3),
    ]  # fmt: skip
    assert labels("generator.pv_delta3") == [("a", "b", r3), ("b", "c", r3), ("c", "a", r3)]
    write_circuit(tmp_path / "src" / "s", SPLIT_PHASE)
    split = prepare_network(Network(id="s", source="src", master="s/master.dss"), root=tmp_path)
    load = next(e for e in split.model.elements if e.name.lower() == "load.l240")
    ((p, q),) = protection_pairs(split.model, load)
    assert {split.model.nodes[p].conductor, split.model.nodes[q].conductor} == {"s1", "s2"}
    assert pair_base_v(split.model, p, q) == pytest.approx(240.0)


def test_a_unit_trips_beyond_either_threshold_on_any_pair(tmp_path: Path) -> None:
    r = runner(tmp_path, config())
    assert r.trip is not None
    assert r.pv is not None
    angle = {"a": 0.0, "b": -2 * np.pi / 3, "c": 2 * np.pi / 3}
    v = np.array(
        [
            0.0 if n.conductor == "n" else n.nominal_v_ln * np.exp(1j * angle[n.conductor])
            for n in r.model.nodes
        ]
    )
    n_units = len(r.pv.units)
    for factor, trips in ((1.0, False), (1.09, False), (1.11, True), (0.71, False), (0.69, True)):
        assert r.trip.beyond(factor * v).tolist() == [trips] * n_units, factor
    elements = {e.name.lower(): e for e in r.model.elements}
    first = protection_pairs(r.model, elements[r.pv.units[0].generator.lower()])[0][0]
    one = v.copy()
    one[first] *= 1.2  # only the first unit reads this node
    assert r.trip.beyond(one).tolist() == [True] + [False] * (n_units - 1)


# ------------------------------------------------------------------- the loop


@pytest.mark.parametrize("v_src", [1.0, 1.12])
def test_raising_the_voltage_trips_units(tmp_path: Path, v_src: float) -> None:
    """At 1.12 pu at the source every unit is above OV 1.10 whatever its output: all trip, the
    label holds them at zero and converges; at 1.0 none does."""
    r = runner(tmp_path, config(scenarios={"source_voltage_pu": v_src}))
    assert r.trip is not None
    for s in range(4):
        o = r.run(s)
        assert o.pf_status == "converged", o.message
        assert o.pv_tripped is not None
        assert o.pv_p_w is not None
        assert o.pv_q_var is not None
        assert o.pv_tripped.all() == (v_src > 1.1)
        assert o.pv_tripped.any() == (v_src > 1.1)
        assert np.all(o.pv_p_w[o.pv_tripped] == 0)
        assert np.all(o.pv_q_var[o.pv_tripped] == 0)
        assert o.v is not None
        assert not (r.trip.beyond(o.v) & ~o.pv_tripped).any()


def test_tripped_units_stay_tripped(tmp_path: Path) -> None:
    """Units far out push their voltage above OV; once they trip it falls back inside, and they
    stay out: in the label a tripped unit can read a voltage within its thresholds."""
    cfg = config(
        der={"pv": {"penetration": 1.0, "kwp": 30.0, "trip": TRIP}},
        scenarios={"source_voltage_pu": 1.04, "n_scenarios_per_variant": 40},
    )
    r = runner(tmp_path, cfg)
    assert r.trip is not None
    latched = 0
    for s in range(40):
        o = r.run(s)
        assert o.pf_status == "converged", o.message
        assert o.pv_tripped is not None
        assert o.v is not None
        inside = ~r.trip.beyond(o.v)
        assert not (~inside & ~o.pv_tripped).any()  # no unit in service beyond a threshold
        latched += int((o.pv_tripped & inside).sum())
    assert latched > 0


def test_the_label_is_the_engine_state_with_the_tripped_units_at_zero(tmp_path: Path) -> None:
    r = runner(tmp_path, config(scenarios={"source_voltage_pu": 1.12}))
    assert r.pv is not None
    o = r.run(0)
    assert o.pv_tripped is not None
    assert o.pv_tripped.all()
    api = r.ctx.api
    for unit in r.pv.units:
        api.Circuit.SetActiveElement(unit.generator)
        assert float(np.abs(np.asarray(api.CktElement.Powers())).sum()) == 0.0
    assert o.n_off_setpoint == 0


def test_the_trip_is_deterministic_and_order_free(tmp_path: Path) -> None:
    cfg = config(
        der={"pv": {"penetration": 1.0, "kwp": 30.0, "trip": TRIP}},
        scenarios={"source_voltage_pu": 1.04, "n_scenarios_per_variant": 12},
    )
    a = [runner(tmp_path / "a", cfg).run(s) for s in range(12)]
    rb = runner(tmp_path / "b", cfg)
    b = {s: rb.run(s) for s in reversed(range(12))}
    assert any(o.pv_tripped is not None and o.pv_tripped.any() for o in a)
    for s, o in enumerate(a):
        assert o.pv_tripped is not None
        assert o.v is not None
        assert np.array_equal(o.pv_tripped, b[s].pv_tripped)  # type: ignore[arg-type]
        assert np.array_equal(o.v, b[s].v)  # type: ignore[arg-type]


def test_unreachable_thresholds_change_no_voltage(tmp_path: Path) -> None:
    """Enabled but never beyond: every outcome bit-equal to the run without trip."""
    pv = {"penetration": 1.0, "kwp": 30.0}
    scenarios = {"source_voltage_pu": 1.04}
    off = runner(tmp_path / "off", config(der={"pv": pv}, scenarios=scenarios))
    wide = {"enabled": True, "ov_pu": 100.0, "uv_pu": 0.01}
    on = runner(tmp_path / "on", config(der={"pv": {**pv, "trip": wide}}, scenarios=scenarios))
    assert off.trip is None
    assert on.trip is not None
    for s in range(8):
        a, b = off.run(s), on.run(s)
        assert a.pv_tripped is None
        assert b.pv_tripped is not None
        assert not b.pv_tripped.any()
        assert a.pf_status == b.pf_status == "converged"
        assert np.array_equal(a.v, b.v)  # type: ignore[arg-type]
        assert np.array_equal(a.pv_p_w, b.pv_p_w)  # type: ignore[arg-type]


def test_a_pass_that_fails_after_a_trip_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solve that does not converge after units tripped: not_converged, with how many tripped."""
    r = runner(tmp_path, config(scenarios={"source_voltage_pu": 1.12}))
    solve = r._solve
    calls = []

    def second_fails() -> SolveStatus:
        calls.append(1)
        status = solve()
        return status if len(calls) == 1 else SolveStatus(False, status.iterations)

    monkeypatch.setattr(r, "_solve", second_fails)
    o = r.run(0)
    assert o.pf_status == "not_converged"
    assert o.pv_tripped is not None
    assert o.pv_tripped.all()
    assert o.message is not None
    assert "after tripping 3 PV unit(s)" in o.message
    assert o.state_id is not None
    assert r.registry.counts[o.state_id] == 1


# ------------------------------------------------------------------- draws and config


def test_thresholds_are_drawn_per_variant_from_their_own_stream(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", VARIANT_FEEDER)
    targets = collect_targets(
        prepare_network(Network(id="t", source="src", master="master.dss"), root=tmp_path)
    )
    pv = PV.model_validate({"penetration": 0.5})

    def draw(k: int, trip: Any = None) -> Any:
        return sample_perturbation(Variants(), targets, SEED, 3, k, der=pv, pv_trip=trip)

    union = (TRIP_OV_UNION, TRIP_UV_UNION)
    without = draw(0)
    with_trip = draw(0, union)
    assert without.pv_trip is None
    assert "pv_trip" not in json.loads(without.to_json())
    assert with_trip.pv_kwp == without.pv_kwp  # nothing else moves
    assert json.loads(with_trip.to_json())["pv_trip"] == dict(
        zip(("ov_pu", "uv_pu"), with_trip.pv_trip, strict=True)
    )
    drawn = [draw(k, union).pv_trip for k in range(400)]
    ov = Counter(d[0] for d in drawn)
    uv = Counter(d[1] for d in drawn)
    assert set(ov) == {1.10, 1.14}
    assert set(uv) == {0.70, 0.80, 0.85, 0.88}
    assert all(abs(c / 400 - 0.5) < 0.1 for c in ov.values())
    assert all(abs(c / 400 - 0.25) < 0.08 for c in uv.values())
    # a fixed OV draws nothing and leaves the UV draws as they are
    fixed = [draw(k, (Fixed(1.12), TRIP_UV_UNION)).pv_trip for k in range(50)]
    assert [f[1] for f in fixed] == [d[1] for d in drawn[:50]]
    assert {f[0] for f in fixed} == {1.12}
    us = Counter(draw(k, TRIP_NATIONAL["US"]).pv_trip[1] for k in range(600))
    assert set(us) == {0.70, 0.88}
    assert abs(us[0.70] / 600 - 2 / 3) < 0.06


def test_national_settings_by_country() -> None:
    trip = PVTrip.model_validate({"enabled": True, "mode": "national"})
    assert trip.settings("GB") == (Fixed(1.14), Fixed(0.80))
    ov, uv = trip.settings("US")
    assert ov == Fixed(1.10)
    assert uv.support() == (0.70, 0.88)
    with pytest.raises(ValueError, match="no public national setpoint was found for Portugal"):
        trip.settings("PT")
    union = PVTrip.model_validate({"enabled": True})
    assert union.settings("PT") == (TRIP_OV_UNION, TRIP_UV_UNION)


def test_national_on_a_portuguese_network_is_a_config_error() -> None:
    data = {
        **BASE, "networks": {"include": ["edp/rural", "ieee13"]},
        "der": {"pv": {"trip": {"enabled": True, "mode": "national"}}},
    }  # fmt: skip
    with pytest.raises(ConfigError, match=r"der\.pv\.trip\.mode: edp/rural: .*Portugal"):
        validate_config(data)
    data["networks"] = {"include": ["ieee13", "csiro_enwl/n01_f1"]}
    assert validate_config(data).der.pv.trip.mode == "national"
    data["der"] = {"pv": {"trip": {"enabled": True}}}  # union: any country
    data["networks"] = {"include": ["edp/rural"]}
    validate_config(data)


def test_generate_refuses_national_without_settings(tmp_path: Path) -> None:
    """The executor checks too: ``generate`` is called without ``load_config`` in tests and
    scripts."""
    cfg = config(der={"pv": {"trip": {"enabled": True, "mode": "national"}}})
    with pytest.raises(GenerationError, match=r"t/lv: der\.pv\.trip mode 'national': .*Portugal"):
        jobs_of(cfg, catalog(tmp_path, "PT"))
    assert [j.pv_trip for j in jobs_of(cfg, catalog(tmp_path, "GB"))] == [TRIP_NATIONAL["GB"]]


@pytest.mark.parametrize(
    ("trip", "match"),
    [
        ({"mode": "national", "ov_pu": 1.2}, "mode 'national' takes the country's settings"),
        ({"uv_pu": 1.12}, "every uv_pu value must lie below every ov_pu value"),
    ],
)
def test_trip_config_errors(trip: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PVTrip.model_validate({"enabled": True, **trip})


def test_trip_and_opf_labels_are_refused_together() -> None:
    with pytest.raises(ValueError, match=r"der\.pv\.trip and labels\.opf cannot both be enabled"):
        config(labels={"opf": {"enabled": True}})


def test_a_stored_config_reloads() -> None:
    cfg = config(der={"pv": {"trip": {"enabled": True, "mode": "national"}}})
    assert GenerationConfig.model_validate(cfg.model_dump(mode="json")) == cfg


# ------------------------------------------------------------------- the dataset


@pytest.mark.parametrize("enabled", [False, True])
def test_the_dataset_records_the_trip(tmp_path: Path, enabled: bool) -> None:
    trip = {"enabled": True, "mode": "national"} if enabled else {"enabled": False}
    cfg = config(
        networks={"include": ["t/*"]},
        der={"pv": {"penetration": 0.6, "kwp": 30.0, "trip": trip}, "storage": {"share": 0.5}},
        scenarios={"source_voltage_pu": {"distribution": "uniform", "low": 1.0, "high": 1.12}},
        settings={"seed": SEED, "num_processes": 1},
    )
    out = tmp_path / "ds"
    manifest = generate(cfg, out, catalog=catalog(tmp_path / "cat", "GB"), root=tmp_path / "cat")
    der = pq.read_table(out / "scenarios" / "der").to_pylist()
    meta = pq.read_table(out / "scenarios" / "scenario_meta").to_pylist()
    variants = pq.read_table(out / "variants" / "variants").to_pylist()
    passport = json.loads((out / "catalog" / "passports" / "t" / "lv.json").read_text())
    assert passport["country"] == "GB"
    if not enabled:
        assert all(d["tripped"] is None for d in der)
        assert all(m["n_pv_tripped"] is None for m in meta)
        assert all("pv_trip" not in json.loads(v["perturbation"]) for v in variants)
        assert "pv_trip" not in passport
        assert all("pv_trip" not in n for n in manifest.networks)
        return
    record = {"mode": "national", "country": "GB", "ov_pu": 1.14, "uv_pu": 0.8}
    assert passport["pv_trip"] == record
    assert all(n["pv_trip"] == record for n in manifest.networks)
    assert all(json.loads(v["perturbation"])["pv_trip"] == {"ov_pu": 1.14, "uv_pu": 0.8}
               for v in variants)  # fmt: skip
    assert {d["tripped"] is None for d in der if d["der_type"] == "bess"} == {True}
    pv = [d for d in der if d["der_type"] == "pv"]
    assert {d["tripped"] for d in pv} == {False, True}
    assert all(d["p_w"] == d["q_var"] == 0.0 for d in pv if d["tripped"])
    per_scenario = Counter(d["scenario_id"] for d in pv if d["tripped"])
    assert {m["scenario_id"]: m["n_pv_tripped"] for m in meta} == {
        m["scenario_id"]: per_scenario.get(m["scenario_id"], 0) for m in meta
    }
