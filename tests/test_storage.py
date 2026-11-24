"""Tests for storage units: placement per variant, fixed set-points per scenario."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import (
    PV,
    GenerationConfig,
    Storage,
    StorageProfile,
    Variants,
)
from gridfm_datakit_distr.config.value_spec import Fixed
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.scenarios.der import StorageSetpoints, storage_units
from gridfm_datakit_distr.scenarios.seeds import VariantStream, variant_rng
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.solve.tables import chunk_tables
from gridfm_datakit_distr.variants import perturbation
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.perturbation import (
    edit_script,
    sample_perturbation,
    storage_kw_for_pv,
)
from gridfm_datakit_distr.variants.targets import VariantTargets, collect_targets
from tests.circuits import VARIANT_FEEDER, write_circuit

SEED = 5
HALF_PV = PV.model_validate({"penetration": 0.5})
ALL_PV = PV.model_validate({"penetration": 1.0})


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Network, Path, PreparedNetwork]:
    root = tmp_path_factory.mktemp("storage")
    write_circuit(root / "src", VARIANT_FEEDER)
    network = Network(id="t/storage", source="src", master="master.dss")
    return network, root, prepare_network(network, root=root)


@pytest.fixture(scope="module")
def targets(base: tuple[Network, Path, PreparedNetwork]) -> VariantTargets:
    return collect_targets(base[2])


def storage(**changes: Any) -> Storage:
    return Storage.model_validate({"share": 1.0, **changes})


# ------------------------------------------------------------------- placement


def test_the_default_places_nothing(targets: VariantTargets) -> None:
    assert Storage().share == Fixed(0.0)
    without = sample_perturbation(Variants(), targets, SEED, 0, 0, der=HALF_PV)
    default = sample_perturbation(Variants(), targets, SEED, 0, 0, der=HALF_PV, storage=Storage())
    assert default == without
    assert default.to_json() == without.to_json()
    assert edit_script(default, targets) == edit_script(without, targets)


@pytest.mark.parametrize("kw", [None, {"distribution": "uniform", "low": 3.0, "high": 5.0}])
def test_placement_is_deterministic_and_monotone_in_the_share(
    targets: VariantTargets, kw: dict[str, Any] | None
) -> None:
    def placed(share: float, pv: PV = HALF_PV) -> dict[str, float]:
        spec = storage(share=share, kw=kw)
        return sample_perturbation(Variants(), targets, SEED, 0, 0, der=pv, storage=spec).storage_kw

    kwp = sample_perturbation(Variants(), targets, SEED, 0, 0, der=HALF_PV).pv_kwp
    assert 0 < len(kwp) < 6
    assert placed(0.0) == {}
    sets = [set(placed(share)) for share in (0.2, 0.5, 0.8, 1.0)]
    assert all(a <= b for a, b in itertools.pairwise(sets))  # common random numbers
    assert sets[-1] == set(kwp)  # storage only with PV, every PV customer at share 1
    full = placed(1.0)
    if kw is None:
        assert full == {e: storage_kw_for_pv(kwp[e]) for e in full}  # the default rating
    else:
        assert all(3.0 <= r < 5.0 for r in full.values())  # the drawn rating
    half = placed(0.5)
    assert all(half[k] == full[k] for k in half)  # ratings do not depend on who is chosen
    assert placed(0.5) == half  # deterministic
    assert set(placed(1.0, ALL_PV)) == {c.element for c in targets.customers}


# ------------------------------------------------------------------- rating


@pytest.mark.parametrize(
    ("kwp", "kw"),
    [
        (0.5, 4.0),
        (3.99, 4.0),
        (4.0, 8.0),
        (9.99, 8.0),
        (10.0, 25.0),
        (149.99, 25.0),
        (150.0, 100.0),
        (1000.0, 100.0),
    ],
)
def test_the_smart_ds_table_maps_pv_size_to_rating(kwp: float, kw: float) -> None:
    """Our reading of the source's bands < 4, 4-10, 10-150, > 150: each lower bound included."""
    assert storage_kw_for_pv(kwp) == kw


def test_the_default_rating_follows_the_customers_pv(
    targets: VariantTargets, monkeypatch: pytest.MonkeyPatch
) -> None:
    streams: list[VariantStream] = []

    def recording(seed: int, net: int, k: int, stream: VariantStream, attempt: int = 0) -> Any:
        streams.append(stream)
        return variant_rng(seed, net, k, stream, attempt)

    monkeypatch.setattr(perturbation, "variant_rng", recording)
    assert Storage().kw is None
    p = sample_perturbation(Variants(), targets, SEED, 0, 0, der=ALL_PV, storage=storage())
    assert len(p.storage_kw) == 6
    assert p.storage_kw == {e: storage_kw_for_pv(p.pv_kwp[e]) for e in p.storage_kw}
    assert set(p.storage_kw.values()) == {4.0, 8.0}  # PV of 2-6 kWp: two bands
    assert VariantStream.STORAGE_PLACEMENT in streams
    assert VariantStream.STORAGE_SIZE not in streams  # no draw for the rating


def test_an_explicit_rating_is_drawn_per_unit_as_before(targets: VariantTargets) -> None:
    spec = storage(kw={"distribution": "uniform", "low": 3.0, "high": 5.0})
    p = sample_perturbation(Variants(), targets, SEED, 0, 0, der=ALL_PV, storage=spec)
    assert p.storage_kw == {  # the ratings of the former default U(3, 5), bit for bit
        "Load.h1": 4.466401725159107,
        "Load.h2": 3.5716675664249493,
        "Load.h3": 3.999613081564434,
        "Load.h4": 3.4203961144864015,
        "Load.h5": 3.349082499193122,
        "Load.pp": 4.745911618975316,
    }


def test_units_are_generators_on_their_customers_nodes(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    variants = Variants.model_validate({"phase_reallocation": {"enabled": True, "fraction": 1.0}})
    result = generate_variant(
        network, targets, variants, SEED, 0, root=root, der=ALL_PV, storage=storage()
    )
    assert result.status == "ok", result.failures  # the read-back checks the nodes
    p = result.perturbation
    assert p is not None
    assert len(p.storage_kw) == 6
    news = [c for c in result.script if c.startswith("New Generator.bess_")]
    assert len(news) == 6
    assert all(" kW=0 kvar=0 model=1 " in c for c in news)
    assert '"storage_kw"' in p.to_json()
    units = storage_units(p)
    assert [u.generator for u in units] == sorted(u.generator for u in units)


# ------------------------------------------------------------------- set-points


def test_setpoints_are_within_the_rating_and_drawn_per_unit(targets: VariantTargets) -> None:
    p = sample_perturbation(Variants(), targets, SEED, 0, 0, der=ALL_PV, storage=storage())
    fleet = StorageSetpoints(storage_units(p), StorageProfile())
    rated = np.array([1e3 * u.kw for u in fleet.units])
    draws = np.array([fleet.draw(SEED, 0, 0, s) for s in range(20)])
    assert (np.abs(draws) <= rated).all()
    assert (draws > 0).any()
    assert (draws < 0).any()  # charging and discharging
    assert np.array_equal(draws[3], fleet.draw(SEED, 0, 0, 3))  # deterministic
    assert len(set(draws[0].round(6))) == len(fleet.units)  # per unit
    fixed = StorageSetpoints(fleet.units, StorageProfile(setpoint=-0.5))
    assert np.array_equal(fixed.draw(SEED, 0, 0, 7), -0.5 * rated)


def test_a_setpoint_outside_minus_one_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="<= 1"):
        StorageProfile.model_validate(
            {"setpoint": {"distribution": "uniform", "low": 0, "high": 2}}
        )
    with pytest.raises(ValueError, match=">= -1"):
        StorageProfile.model_validate({"setpoint": -1.5})


# ------------------------------------------------------------------- power flow


def runner(tmp_path: Path, **sections: Any) -> VariantRunner:
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
    write_circuit(tmp_path / "src", VARIANT_FEEDER)
    network = Network(id="t/net", source="src", master="master.dss")
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options, tmp_path))
    result = generate_variant(
        network, targets, cfg.variants, SEED, 0, options=options, root=tmp_path,
        der=cfg.der.pv, storage=cfg.der.storage,
    )  # fmt: skip
    assert result.status == "ok"
    return VariantRunner(network, result, cfg, 0, root=tmp_path)


WITH_STORAGE: dict[str, Any] = {
    "der": {"pv": {"penetration": 1.0}, "storage": {"share": 1.0}},
}


def test_the_engine_delivers_the_sampled_setpoints(tmp_path: Path) -> None:
    r = runner(tmp_path, **WITH_STORAGE)
    assert r.storage is not None
    assert len(r.storage.units) == 6
    api = r.ctx.api
    for s in range(4):
        o = r.run(s)
        assert o.pf_status == "converged", o.message  # the setpoint check covers them
        assert o.residual_rel is not None
        assert o.residual_rel < 1e-6  # Kirchhoff on the extracted admittance, storage present
        assert o.bess_p_w is not None
        assert np.array_equal(o.bess_p_w, r.storage.draw(SEED, r.net, r.k, s))
        for unit, p_w in zip(r.storage.units, o.bess_p_w, strict=True):
            api.Circuit.SetActiveElement(unit.generator)
            delivered = -float(np.asarray(api.CktElement.Powers())[0::2].sum()) * 1e3
            assert delivered == pytest.approx(p_w, rel=1e-6, abs=1e-6)
            assert float(api.Generators.kvar()) == 0.0


def test_a_unit_off_its_setpoint_is_caught(tmp_path: Path) -> None:
    r = runner(tmp_path, **WITH_STORAGE)
    assert r.storage is not None
    first = r.storage.units[0].generator
    r.ctx.command(f"Edit {first} vminpu=1.2")  # outside its window: constant impedance
    o = r.run(0)
    assert o.pf_status == "off_setpoint"
    assert o.n_off_setpoint >= 1


def test_storage_rows_in_the_der_table(tmp_path: Path) -> None:
    r = runner(tmp_path, **WITH_STORAGE)
    assert r.pv is not None
    assert r.storage is not None
    outcomes = [r.run(s) for s in range(3)]
    tables = chunk_tables(
        r, outcomes, precision="full", config_hash="h", versions={"generator": "x"}
    )
    der = tables["scenarios/der"].to_pylist()
    n_pv, n_bess = len(r.pv.units), len(r.storage.units)
    assert len(der) == 3 * (n_pv + n_bess)
    first = der[: n_pv + n_bess]
    assert [d["der_type"] for d in first] == ["pv"] * n_pv + ["bess"] * n_bess
    index = {e.name: e.idx for e in r.model.elements}
    bess = first[n_pv:]
    assert [d["element_idx"] for d in bess] == [index[u.generator] for u in r.storage.units]
    o = outcomes[0]
    assert o.bess_p_w is not None
    assert [d["p_w"] for d in bess] == o.bess_p_w.tolist()
    assert all(d["p_avail_w"] == d["p_w"] and d["q_var"] == 0.0 for d in bess)
    assert [d["s_rated_va"] for d in bess] == [1e3 * u.kw for u in r.storage.units]


def test_the_load_ramp_scales_storage_too(tmp_path: Path) -> None:
    from gridfm_datakit_distr.solve.pf_batch import _Draws, _scaled

    draws = _Draws(np.ones(2), np.ones(2), None, None, 1.0, np.array([2.0, -4.0]))
    assert _scaled(draws, 0.5).bess is not None
    assert np.array_equal(_scaled(draws, 0.5).bess, [1.0, -2.0])  # type: ignore[arg-type]
