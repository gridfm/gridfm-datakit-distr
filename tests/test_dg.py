"""Tests for dispatchable generators: sites and capacity per network, set-points per scenario."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import DG, DGProfile, GenerationConfig, Variants
from gridfm_datakit_distr.config.value_spec import Fixed
from gridfm_datakit_distr.io.manifest import Manifest
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.opf.support import vpn_bounds
from gridfm_datakit_distr.scenarios.der import DGSetpoints, dg_units
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.solve.tables import chunk_tables
from gridfm_datakit_distr.variants.dg import BISECT_REL, SCAN_STEPS, dg_capacity, search
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.perturbation import (
    Perturbation,
    dg_sites,
    edit_script,
    sample_perturbation,
)
from gridfm_datakit_distr.variants.targets import VariantTargets, collect_targets
from tests.circuits import DG_FEEDER, OPF_LOADS, write_circuit

SEED = 5


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Network, Path, PreparedNetwork]:
    root = tmp_path_factory.mktemp("dg")
    write_circuit(root / "src", DG_FEEDER)
    network = Network(id="t/dg", source="src", master="master.dss")
    return network, root, prepare_network(network, root=root)


@pytest.fixture(scope="module")
def targets(base: tuple[Network, Path, PreparedNetwork]) -> VariantTargets:
    return collect_targets(base[2])


# ------------------------------------------------------------------- sites


def test_sites_are_every_fourth_customer_in_name_order(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    """The 4th, 8th, ... enabled load in name order (a string sort: h1, h10, h2, ...), not in the
    engine's order nor in natural order."""
    engine = [f"Load.{n}" for n in base[2].ctx.api.Loads.AllNames()]
    assert engine[3::4] == ["Load.h2", "Load.h4"]  # the circuit defines them out of order
    assert [c.element for c in dg_sites(targets)] == ["Load.h3", "Load.h7"]


def test_the_default_places_nothing(targets: VariantTargets) -> None:
    assert DG().enabled is False
    without = sample_perturbation(Variants(), targets, SEED, 0, 0)
    off = sample_perturbation(Variants(), targets, SEED, 0, 0, dg_kw=None)
    assert off == without
    assert off.to_json() == without.to_json()
    assert edit_script(off, targets) == edit_script(without, targets)


def test_units_sit_on_their_loads_nodes_after_a_phase_move(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    variants = Variants.model_validate({"phase_reallocation": {"enabled": True, "fraction": 1.0}})
    result = generate_variant(network, targets, variants, SEED, 0, root=root, dg_kw=12.5)
    assert result.status == "ok", result.failures  # the read-back checks the nodes
    p = result.perturbation
    assert p is not None
    assert p.dg_kw == {"Load.h3": 12.5, "Load.h7": 12.5}
    news = {c.split()[1]: c for c in result.script if c.startswith("New Generator.dg_")}
    assert sorted(news) == ["Generator.dg_h3", "Generator.dg_h7"]
    for load in ("Load.h3", "Load.h7"):
        moved = p.phases[load][1]  # every movable load moves at fraction 1
        assert f" bus1={moved} " in news[f"Generator.dg_{load.split('.')[1]}"]
    assert all(" kW=0 kvar=0 model=1 " in c for c in news.values())
    assert '"dg_kw"' in p.to_json()
    units = dg_units(p)
    assert [(u.generator, u.load, u.kw) for u in units] == [
        ("Generator.dg_h3", "Load.h3", 12.5),
        ("Generator.dg_h7", "Load.h7", 12.5),
    ]


# ------------------------------------------------------------------- capacity search


def test_the_search_brackets_the_crossing_of_a_flow_that_falls_then_rises() -> None:
    probes: list[float] = []

    def probe(p: float) -> tuple[bool, float]:
        probes.append(p)
        return True, abs(p - 3.0) / 4.3  # 0.70 at 0, 0 at 3, 1 at 7.3

    status, lo, hi = search(probe, 1.0)
    assert status == "ok"
    assert probes[:4] == [1.0, 2.0, 4.0, 8.0]  # the scan stops at the first point at a bound
    assert lo < 7.3 <= hi
    assert hi - lo <= BISECT_REL * hi


def test_the_search_reports_divergence_and_no_bound() -> None:
    status, _, hi = search(lambda p: (p < 5.0, 0.1 * p if p < 5.0 else math.nan), 1.0)
    assert status == "not_converged"  # diverged at 8 before any bound
    assert math.isnan(hi)
    probes: list[float] = []

    def flat(p: float) -> tuple[bool, float]:
        probes.append(p)
        return True, 0.5

    status, _, _ = search(flat, 2.0)
    assert status == "no_binding_limit"
    assert probes == [2.0 * 2**k for k in range(SCAN_STEPS)]


# ------------------------------------------------------------------- capacity on a feeder


def max_voltage_ratio(network: Network, root: Path, targets: VariantTargets, p_w: float) -> float:
    """The largest phase-to-neutral voltage over the OPF's upper bound (1.10 pu), every DG at
    ``p_w``, solved on its own compiled network."""
    kw = {c.element: 0.0 for c in dg_sites(targets)}
    script = edit_script(Perturbation(0, dg_kw=kw), targets)
    prepared = prepare_network(network, root=root, edits=script)
    api = prepared.ctx.api
    for c in dg_sites(targets):
        api.Generators.Name(f"dg_{c.name}")
        api.Generators.kW(p_w / 1e3)
    assert prepared.ctx.solve().converged
    model = prepared.model
    pairs, _, hi = vpn_bounds(model, 0.94, 1.10)
    positions = prepared.ctx.ynode_positions(n.opendss_node for n in model.nodes)
    v = prepared.ctx.ynode_voltages()[positions]
    vn = np.where(pairs[:, 1] == GROUND, 0.0, v[pairs[:, 1]])
    return float(np.max(np.abs(v[pairs[:, 0]] - vn) / hi))


def test_the_capacity_is_the_smallest_that_reaches_the_voltage_bound(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    cap = dg_capacity(network, targets, 1.10, root=root)
    assert (cap.status, cap.binding, cap.n_units, cap.ignored) == ("ok", "voltage", 2, ())
    assert cap.p_max_w is not None
    assert cap.kw == cap.p_max_w / 1e3
    assert max_voltage_ratio(network, root, targets, cap.p_max_w) >= 1.0
    assert max_voltage_ratio(network, root, targets, cap.p_max_w * (1 - 2 * BISECT_REL)) < 1.0
    assert cap.record() == {
        "status": "ok", "n_units": 2, "p_max_w": cap.p_max_w, "binding": "voltage",
        "ignored": [],
    }  # fmt: skip


def test_a_transformer_rating_binds_when_the_voltage_bound_is_far(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    cap = dg_capacity(network, targets, 1.40, root=root)
    assert (cap.status, cap.binding) == ("ok", "Transformer.t")


def test_a_transformer_above_its_rating_without_dg_is_not_counted(tmp_path: Path) -> None:
    small = {"master.dss": DG_FEEDER["master.dss"].replace("kVAs=[100 100]", "kVAs=[20 20]")}
    write_circuit(tmp_path / "src", small)
    network = Network(id="t/small", source="src", master="master.dss")
    targets = collect_targets(prepare_network(network, root=tmp_path))
    cap = dg_capacity(network, targets, 1.10, root=tmp_path)
    assert (cap.status, cap.binding, cap.ignored) == ("ok", "voltage", ("Transformer.t",))


def test_networks_without_a_capacity_get_a_status(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets, tmp_path: Path
) -> None:
    network, root, _ = base
    at_bound = dg_capacity(network, targets, 0.95, root=root)  # 1.0 pu exceeds it without DG
    assert (at_bound.status, at_bound.p_max_w, at_bound.kw) == ("voltage_at_bound", None, None)
    write_circuit(tmp_path / "src", OPF_LOADS)
    three = Network(id="t/three", source="src", master="master.dss")
    three_targets = collect_targets(prepare_network(three, root=tmp_path))
    cap = dg_capacity(three, three_targets, 1.1, root=tmp_path)
    assert (cap.status, cap.n_units, cap.p_max_w) == ("no_site", 0, None)


# ------------------------------------------------------------------- set-points


def test_setpoints_are_within_the_capacity_and_drawn_per_unit() -> None:
    p = Perturbation(0, dg_kw={"Load.a": 10.0, "Load.b": 10.0, "Load.c": 10.0})
    fleet = DGSetpoints(dg_units(p), DGProfile())
    draws = np.array([fleet.draw(SEED, 0, 0, s) for s in range(20)])
    assert (draws >= 0).all()
    assert (draws <= 10e3).all()
    assert np.array_equal(draws[3], fleet.draw(SEED, 0, 0, 3))  # deterministic
    assert len(set(draws[0].round(6))) == 3  # per unit
    assert DGProfile().setpoint != Fixed(1.0)
    fixed = DGSetpoints(fleet.units, DGProfile(setpoint=0.5))
    assert np.array_equal(fixed.draw(SEED, 0, 0, 7), np.full(3, 5e3))


def test_a_setpoint_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        DGProfile.model_validate({"setpoint": -0.5})


# ------------------------------------------------------------------- power flow


def runner(tmp_path: Path, kw: float = 10.0, **sections: Any) -> VariantRunner:
    cfg = GenerationConfig.model_validate(
        {
            "dataset_name": "t",
            "networks": {"include": ["x"]},
            "scenarios": {"n_scenarios_per_variant": 10},
            "controls": {"mode": "as_is"},
            "settings": {"seed": SEED, "residual_sample_rate": 1.0},
            **sections,
        }
    )
    write_circuit(tmp_path / "src", DG_FEEDER)
    network = Network(id="t/net", source="src", master="master.dss")
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options, tmp_path))
    result = generate_variant(
        network, targets, cfg.variants, SEED, 0, options=options, root=tmp_path,
        der=cfg.der.pv, storage=cfg.der.storage, dg_kw=kw,
    )  # fmt: skip
    assert result.status == "ok"
    return VariantRunner(network, result, cfg, 0, root=tmp_path)


ALL_DER: dict[str, Any] = {"der": {"pv": {"penetration": 1.0}, "storage": {"share": 1.0}}}


def test_the_engine_delivers_the_drawn_setpoints(tmp_path: Path) -> None:
    r = runner(tmp_path)
    assert r.dg is not None
    assert [u.generator for u in r.dg.units] == ["Generator.dg_h3", "Generator.dg_h7"]
    api = r.ctx.api
    for s in range(4):
        o = r.run(s)
        assert o.pf_status == "converged", o.message  # the setpoint check covers them
        assert o.residual_rel is not None
        assert o.residual_rel < 1e-6  # Kirchhoff on the extracted admittance, DG present
        assert o.dg_p_w is not None
        assert np.array_equal(o.dg_p_w, r.dg.draw(SEED, r.net, r.k, s))
        for unit, p_w in zip(r.dg.units, o.dg_p_w, strict=True):
            api.Circuit.SetActiveElement(unit.generator)
            delivered = -float(np.asarray(api.CktElement.Powers())[0::2].sum()) * 1e3
            assert delivered == pytest.approx(p_w, rel=1e-6, abs=1e-6)
            assert float(api.Generators.kvar()) == 0.0


def test_dg_rows_follow_pv_and_storage_in_the_der_table(tmp_path: Path) -> None:
    r = runner(tmp_path, **ALL_DER)
    assert r.pv is not None
    assert r.storage is not None
    assert r.dg is not None
    outcomes = [r.run(s) for s in range(3)]
    tables = chunk_tables(
        r, outcomes, precision="full", config_hash="h", versions={"generator": "x"}
    )
    der = tables["scenarios/der"].to_pylist()
    n_pv, n_bess, n_dg = len(r.pv.units), len(r.storage.units), len(r.dg.units)
    assert len(der) == 3 * (n_pv + n_bess + n_dg)
    first = der[: n_pv + n_bess + n_dg]
    assert [d["der_type"] for d in first] == ["pv"] * n_pv + ["bess"] * n_bess + ["dg"] * n_dg
    index = {e.name: e.idx for e in r.model.elements}
    dg = first[n_pv + n_bess :]
    assert [d["element_idx"] for d in dg] == [index[u.generator] for u in r.dg.units]
    o = outcomes[0]
    assert o.dg_p_w is not None
    assert [d["p_w"] for d in dg] == o.dg_p_w.tolist()
    assert all(d["p_avail_w"] == d["s_rated_va"] == 10e3 and d["q_var"] == 0.0 for d in dg)


def test_the_load_ramp_scales_dg_too() -> None:
    from gridfm_datakit_distr.solve.pf_batch import _Draws, _scaled

    draws = _Draws(np.ones(2), np.ones(2), None, None, 1.0, None, np.array([2.0, 4.0]))
    assert _scaled(draws, 0.5).dg is not None
    assert np.array_equal(_scaled(draws, 0.5).dg, [1.0, 2.0])  # type: ignore[arg-type]


# ------------------------------------------------------------------- generate


def test_a_power_flow_dataset_with_dg_records_the_capacity_without_casadi(tmp_path: Path) -> None:
    """``generate`` with DG and without OPF labels, in a fresh process: CasADi is never
    imported; the DG of the feeder are in ``scenarios/der`` at their network's capacity, which
    the manifest records with its binding limit; the network without a site has none, and its
    status says why."""
    data = {
        "dataset_name": "dg",
        "networks": {"include": ["t/*"]},
        "variants": {"n_variants": 2},
        "der": {"dg": {"enabled": True}},
        "scenarios": {"n_scenarios_per_variant": 3},
        "settings": {"seed": 2, "num_processes": 1},
    }
    out = tmp_path / "out"
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from gridfm_datakit_distr.config.schema import GenerationConfig\n"
        "from gridfm_datakit_distr.solve.executor import generate\n"
        "from tests.circuits import dg_catalog\n"
        "root, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "config = GenerationConfig.model_validate(json.loads(sys.argv[3]))\n"
        "generate(config, out, catalog=dg_catalog(root), root=root)\n"
        "print('casadi' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(out), json.dumps(data)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "False"
    networks = {n["network_id"]: n for n in Manifest.load(out).networks}
    dg = networks["t/dg"]["dg"]
    assert (dg["status"], dg["binding"], dg["n_units"]) == ("ok", "voltage", 2)
    assert dg["p_max_w"] > 0
    assert networks["t/lv"]["dg"]["status"] == "no_site"
    der = pq.read_table(out / "scenarios" / "der").to_pylist()
    rows = [r for r in der if r["der_type"] == "dg"]
    assert len(rows) == 2 * 2 * 3  # two units, two variants, three scenarios
    assert all(r["p_avail_w"] == pytest.approx(dg["p_max_w"], rel=1e-6) for r in rows)
    assert all(0 <= r["p_w"] <= r["p_avail_w"] for r in rows)
    variants = pq.read_table(out / "variants" / "variants").to_pylist()
    kw = [json.loads(v["perturbation"]).get("dg_kw") for v in variants]
    assert sum(k is not None for k in kw) == 2  # both variants of t/dg, none of t/lv
    card = (out / "README.md").read_text()
    assert f"`t/dg` 2 of {dg['p_max_w'] / 1e3:.4g} kW (bound: voltage)" in card
    assert "`t/lv` none (no_site)" in card
