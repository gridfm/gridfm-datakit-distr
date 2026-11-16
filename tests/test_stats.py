"""Tests for the dataset statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.stats.physics import ALPHA, Histogram, physics, report_text, write_report
from gridfm_datakit_distr.stats.plots import write_plots
from tests.circuits import VARIANT_FEEDER
from tests.test_executor import catalog, config


def test_histogram_quantiles() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 0.03, 20_000)
    h = Histogram(0.0, 2.0, 4000)
    h.add(values[:7000])
    h.add(values[7000:])
    width = 2.0 / 4000
    for q in (0.01, 0.5, 0.99):
        assert abs(h.quantile(q) - np.quantile(values, q)) <= width
    assert (h.n, h.minimum, h.maximum) == (20_000, values.min(), values.max())
    assert h.total / h.n == pytest.approx(values.mean())
    h.add(np.array([-1.0, 3.0, np.nan]))  # outside the range: counted, not binned; NaN ignored
    assert h.counts[-2:].tolist() == [1, 1]
    assert h.quantile(0.0) == -1.0
    assert h.quantile(1.0) == 3.0
    assert Histogram(0.0, 1.0, 10).summary() == {"n": 0}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stats")
    out = root / "out"
    generate(config(), out, catalog=catalog(root), root=root)
    return out


def test_physics_against_a_direct_computation(dataset: Path) -> None:
    stats = physics(dataset, 0.94, 1.10)
    assert set(stats) == {"t/feeder", "t/reg"}
    feeder = stats["t/feeder"]
    assert "neutral_v" in feeder.histograms  # the four-wire feeder has neutral nodes
    assert "neutral_v" not in stats["t/reg"].histograms
    # direct computation from the tables of t/feeder
    nodes = pq.read_table(dataset / "variants" / "nodes").to_pylist()
    results = pq.read_table(dataset / "scenarios" / "node_results")
    meta = pq.read_table(dataset / "scenarios" / "scenario_meta").to_pylist()
    variant_of = {m["scenario_id"]: m["variant_id"] for m in meta}
    vm, vuf, neutral = [], [], []
    by_scenario: dict[int, list[complex]] = {}
    for sid, v_re, v_im in zip(
        results.column("scenario_id").to_pylist(),
        results.column("v_re").to_pylist(),
        results.column("v_im").to_pylist(),
        strict=True,
    ):
        by_scenario.setdefault(sid, []).append(complex(v_re, v_im))
    for sid, v in by_scenario.items():
        vid = variant_of[sid]
        if not vid.startswith("t/feeder/"):
            continue
        mine = [n for n in nodes if n["variant_id"] == vid]
        phase: dict[str, dict[str, complex]] = {}
        for n in mine:
            value = v[n["node_idx"]]
            base = n["kv_base_ln"] * 1e3
            if n["conductor"] in "abc":
                vm.append(abs(value) / base)
                phase.setdefault(n["bus"], {})[n["conductor"]] = value
            elif n["conductor"] == "n":
                neutral.append(abs(value) / base)
        for p in phase.values():
            if len(p) == 3:
                v1 = p["a"] + ALPHA * p["b"] + ALPHA**2 * p["c"]
                v2 = p["a"] + ALPHA**2 * p["b"] + ALPHA * p["c"]
                vuf.append(abs(v2) / abs(v1))
    h = feeder.histograms
    assert h["vm_a"].n + h["vm_b"].n + h["vm_c"].n == len(vm)
    assert max(h["vm_a"].maximum, h["vm_b"].maximum, h["vm_c"].maximum) == pytest.approx(max(vm))
    assert (h["vuf"].n, h["vuf"].maximum) == (len(vuf), pytest.approx(max(vuf)))
    assert (h["neutral_v"].n, h["neutral_v"].maximum) == (len(neutral), pytest.approx(max(neutral)))
    assert 0 < h["vuf"].maximum < 0.1


def test_loadings_from_recomputed_currents(tmp_path: Path) -> None:
    """Currents recomputed as Yprim · V give the loadings of the stored element currents."""
    stored = tmp_path / "stored"
    data = config()
    data = data.model_copy(
        update={"labels": data.labels.model_copy(update={"element_currents": True})}
    )
    networks = catalog(tmp_path)
    # the feeder's four-wire linecode declares an ampacity; the regulated network declares none
    master = tmp_path / "src" / "feeder" / "master.dss"
    lc4 = "New Linecode.lc4 nphases=4 units=km"
    master.write_text(VARIANT_FEEDER["master.dss"].replace(lc4, f"{lc4} normamps=200"))
    generate(data, stored, catalog=networks, root=tmp_path)
    currents = pq.read_table(stored / "scenarios" / "element_results")
    stats = physics(stored, 0.94, 1.10)
    assert stats["t/feeder"].histograms["line_loading"].n > 0
    assert stats["t/feeder"].histograms["neutral_loading"].n > 0
    assert "line_loading" not in stats["t/reg"].histograms  # no load against a default ampacity
    # the recomputed currents equal the stored ones (float32 in mixed precision)
    from gridfm_datakit_distr.stats import physics as module

    meta = pq.read_table(stored / "scenarios" / "scenario_meta").to_pylist()
    first = next(m for m in meta if m["pf_status"] == "converged")
    key = "00000-0000"
    single = stored / "states" / "yprim_state.parquet"
    m = module._variant(
        stored, key, first["variant_id"], pq.read_table(single) if single.exists() else None
    )
    plan, _ = module._plan(m, first["state_id"])
    nodes = pq.read_table(stored / "scenarios" / "node_results").to_pylist()
    sid = first["scenario_id"]
    v = np.array([complex(r["v_re"], r["v_im"]) for r in nodes if r["scenario_id"] == sid])
    mine = [r for r in currents.to_pylist() if r["scenario_id"] == first["scenario_id"]]
    stored_i = np.array([complex(r["i_re"], r["i_im"]) for r in mine])
    recomputed = plan.currents(v)
    assert np.abs(recomputed - stored_i).max() <= 1e-6 * np.abs(stored_i).max()


def test_violations_and_report(dataset: Path) -> None:
    every = physics(dataset, 1.5, 2.0)  # every voltage is below the band
    assert all(s.violations["voltage"] == s.scenarios for s in every.values())
    stats = physics(dataset, 0.94, 1.10)
    path = write_report(dataset, stats, 0.94, 1.10)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["voltage_limits_pu"] == [0.94, 1.10]
    feeder = data["networks"]["t/feeder"]
    assert {"vuf", "neutral_v", "vm_a"} <= set(feeder["distributions"])
    # no line declares an ampacity: no neutral rate; the transformer's kVA gives a thermal one
    assert set(feeder["violation_rates"]) == {"voltage", "thermal"}
    assert "t/reg:" in report_text(stats)
    figures = {p.name for p in write_plots(dataset, stats)}
    assert {"voltage.png", "vuf.png", "neutral_voltage.png"} <= figures
