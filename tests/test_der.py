"""Tests for PV placement (per variant) and outputs (per scenario)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import PV, PVProfile, Variants
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.scenarios.der import (
    PVOutputs,
    clear_sky,
    cloud_factor,
    pv_units,
)
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.perturbation import edit_script, sample_perturbation
from gridfm_datakit_distr.variants.targets import VariantTargets, collect_targets
from tests.circuits import VARIANT_FEEDER, write_circuit

SEED = 5
ALL_PV: dict[str, Any] = {
    "penetration": 1.0,
    "kwp": {"distribution": "uniform", "low": 2, "high": 6},
}


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Network, Path, PreparedNetwork]:
    root = tmp_path_factory.mktemp("der")
    write_circuit(root / "src", VARIANT_FEEDER)
    network = Network(id="t/der", source="src", master="master.dss")
    return network, root, prepare_network(network, root=root)


@pytest.fixture(scope="module")
def targets(base: tuple[Network, Path, PreparedNetwork]) -> VariantTargets:
    return collect_targets(base[2])


def pv(**changes: Any) -> PV:
    return PV.model_validate({**ALL_PV, **changes})


# ------------------------------------------------------------------- placement


UNPERTURBED = Variants.model_validate(
    {
        "n_variants": 1,
        "admittance": {"line_length_scale": 1.0},
        "phase_reallocation": {"enabled": False},
    }
)
"""Variants as the defaults were before they perturbed: the source network."""


def test_customers_are_every_enabled_load(targets: VariantTargets) -> None:
    by = {c.element: c for c in targets.customers}
    assert sorted(by) == [f"Load.{n}" for n in ("h1", "h2", "h3", "h4", "h5", "pp")]
    assert (by["Load.h1"].bus1, by["Load.h1"].phases, by["Load.h1"].conn) == ("b1.1.4", 1, "wye")
    assert by["Load.pp"].bus1 == "b3.1.2"


def test_placement_is_monotone_in_the_penetration(targets: VariantTargets) -> None:
    def placed(rho: float) -> dict[str, float]:
        return sample_perturbation(Variants(), targets, SEED, 0, 0, der=pv(penetration=rho)).pv_kwp

    assert placed(0.0) == {}
    sets = [set(placed(rho)) for rho in (0.2, 0.5, 0.8, 1.0)]
    assert all(a <= b for a, b in itertools.pairwise(sets))  # common random numbers
    assert len(sets[-1]) == 6
    full = placed(1.0)
    assert all(2.0 <= kwp < 6.0 for kwp in full.values())
    half = placed(0.5)
    assert all(half[k] == full[k] for k in half)  # sizes do not depend on who is chosen


def test_the_penetration_is_drawn_per_variant(targets: VariantTargets) -> None:
    spec = pv(penetration={"distribution": "uniform", "low": 0.0, "high": 1.0})
    drawn = {
        sample_perturbation(Variants(), targets, SEED, 0, k, der=spec).pv_penetration
        for k in range(5)
    }
    assert len(drawn) == 5


def test_generators_sit_on_their_customers_nodes(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    variants = Variants.model_validate({"phase_reallocation": {"enabled": True, "fraction": 1.0}})
    result = generate_variant(network, targets, variants, SEED, 0, root=root, der=pv())
    assert result.status == "ok", result.failures
    model, p = result.model, result.perturbation
    assert model is not None
    assert p is not None
    elements = {e.name: e for e in model.elements}

    def nodes(name: str) -> list[str]:
        return [
            model.nodes[k].opendss_node if k != GROUND else "0" for k in elements[name].terminals[0]
        ]

    for load in p.pv_kwp:
        generator = f"Generator.pv_{load.split('.', 1)[1]}"
        assert nodes(generator) == nodes(load)
    assert set(p.phases) <= set(p.pv_kwp)  # moved customers keep their PV on the new phase
    news = [c for c in result.script if c.startswith("New Generator.")]
    assert len(news) == 6
    assert all(" kW=0 kvar=0 model=1 vminpu=0.6 vmaxpu=1.5" in c for c in news)


def test_zero_penetration_reproduces_the_base_case(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, prepared = base
    p = sample_perturbation(UNPERTURBED, targets, SEED, 0, 0, der=pv(penetration=0.0))
    assert edit_script(p, targets) == []
    result = generate_variant(
        network, targets, UNPERTURBED, SEED, 0, root=root, der=pv(penetration=0.0)
    )
    assert result.model is not None
    assert [e.name for e in result.model.elements] == [e.name for e in prepared.model.elements]


def test_a_drawn_reactive_limit_is_used(targets: VariantTargets) -> None:
    spec = pv(
        reactive_fraction={"distribution": "uniform", "low": 0.3, "high": 0.5},
        reactive_threshold={"distribution": "uniform", "low": 0.0, "high": 0.1},
    )
    p = sample_perturbation(UNPERTURBED, targets, SEED, 0, 0, der=spec)
    units = pv_units(p, spec)  # does not raise
    assert len(units) == 6
    assert [u.reactive_fraction for u in units] == [p.pv_reactive_fraction[u.load] for u in units]
    assert [u.reactive_threshold for u in units] == [p.pv_reactive_threshold[u.load] for u in units]
    assert len({u.reactive_fraction for u in units}) > 1  # drawn per unit, not the default 0.44


# ------------------------------------------------------------------- outputs


def test_clear_sky_and_clouds() -> None:
    assert clear_sky(12.0) == pytest.approx(1.0)
    assert clear_sky(6.0) == pytest.approx(0.0, abs=1e-15)
    assert clear_sky(0.0) == clear_sky(3.0) == clear_sky(21.0) == 0.0
    assert clear_sky(9.0) == pytest.approx(math.cos(math.pi / 4))
    assert cloud_factor(0.0) == 1.0
    assert cloud_factor(1.0) == pytest.approx(0.25)  # Kasten and Czeplak (1980)
    assert cloud_factor(0.5) == pytest.approx(1 - 0.75 * 0.5**3.4)


def outputs(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets, **changes: Any
) -> tuple[PVOutputs, Any]:
    network, root, _ = base
    spec = pv(**changes)
    result = generate_variant(network, targets, Variants(), SEED, 0, root=root, der=spec)
    assert result.perturbation is not None
    units = pv_units(result.perturbation, spec)
    return PVOutputs(units, spec, PVProfile()), result


def test_outputs_follow_the_profile_and_the_rating(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    fleet, _ = outputs(
        base, targets, power_factor={"distribution": "uniform", "low": 0.9, "high": 1.0}
    )
    kwp = np.array([u.kwp for u in fleet.units])
    rated = np.array([u.s_rated_va for u in fleet.units])
    assert rated == pytest.approx(1100 * kwp)  # 1.1 kVA per kWp, the default
    noon = fleet.draw(SEED, 0, 0, 0, time_s=12 * 3600)
    assert noon.hour == 12.0
    assert noon.p_avail_w == pytest.approx(1000 * kwp * cloud_factor(noon.cloudiness))
    assert np.array_equal(noon.p_w, noon.p_avail_w)
    assert (noon.q_var <= 0).all()  # absorbing below unit power factor
    assert (np.hypot(noon.p_w, noon.q_var) <= rated * (1 + 1e-12)).all()
    night = fleet.draw(SEED, 0, 0, 0, time_s=86400 + 2 * 3600)  # 02:00 on a later day
    assert night.hour == 2.0
    assert not night.p_w.any()
    free = [fleet.draw(SEED, 0, 0, s, time_s=None).hour for s in range(20)]  # synthetic loads
    assert len(set(free)) == 20
    assert all(0 <= h < 24 for h in free)


def test_an_undersized_inverter_clips_the_output(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    fleet, _ = outputs(base, targets, inverter_kva_per_kwp=0.8)  # DC/AC ratio 1.25
    rated = np.array([u.s_rated_va for u in fleet.units])
    draws = [fleet.draw(SEED, 0, 0, s, time_s=12 * 3600) for s in range(10)]
    p_avail, p = (np.concatenate([getattr(d, f) for d in draws]) for f in ("p_avail_w", "p_w"))
    assert (p_avail > np.tile(rated, 10)).any()  # some clipping happens at noon
    assert np.array_equal(p, np.minimum(p_avail, np.tile(rated, 10)))
    for d in draws:
        assert (np.hypot(d.p_w, d.q_var) <= rated * (1 + 1e-12)).all()


def test_the_engine_delivers_the_sampled_outputs(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    fleet, result = outputs(base, targets)
    draw = fleet.draw(SEED, 0, 0, 3, time_s=11 * 3600)
    prepared = prepare_network(
        network, root=root, variant_id=result.variant_id, edits=result.script
    )
    for unit, p_w, q_var in zip(fleet.units, draw.p_w, draw.q_var, strict=True):
        prepared.ctx.command(
            f"Edit {unit.generator} kW={float(p_w) / 1e3!r} kvar={float(q_var) / 1e3!r}"
        )
    assert prepared.ctx.solve().converged
    report = setpoint_report(prepared.ctx)
    assert report.passes()
    api = prepared.ctx.api
    delivered = 0.0
    for unit in fleet.units:
        api.Circuit.SetActiveElement(unit.generator)
        delivered -= float(np.asarray(api.CktElement.Powers())[0::2].sum()) * 1e3
    assert delivered == pytest.approx(draw.p_w.sum(), rel=1e-9)
