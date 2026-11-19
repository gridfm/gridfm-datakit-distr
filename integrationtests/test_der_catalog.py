"""PV units on a catalog network."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import PV, PVProfile, SyntheticProfiles, Variants
from gridfm_datakit_distr.extract.injections import injections
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.scenarios.der import PVOutputs, pv_units
from gridfm_datakit_distr.scenarios.profiles import customers_of
from gridfm_datakit_distr.scenarios.profiles.synthetic import SyntheticProvider
from gridfm_datakit_distr.validate.residual import residual
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]

SPEC = PV.model_validate(
    {"penetration": 0.5, "kwp": {"distribution": "uniform", "low": 2.0, "high": 6.0}}
)


@pytest.mark.parametrize(
    "network_id", ["csiro_enwl/n01_f1", pytest.param("edp/rural", marks=pytest.mark.slow)]
)
def test_pv_units_deliver_their_sampled_outputs(
    network_master: Callable[[str], Path], network_id: str
) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    variants = Variants.model_validate(
        {"n_variants": 2, "phase_reallocation": {"enabled": True, "fraction": 0.2}}
    )
    targets = collect_targets(prepare_network(network))
    results = [
        generate_variant(network, targets, variants, 11, k, der=SPEC)
        for k in range(variants.n_variants)
    ]
    for r in results:
        assert r.status == "ok", r.failures
        assert r.perturbation is not None
        assert r.model is not None
        n_customers = sum(e.element_class == "load" for e in r.model.elements)
        assert 0.35 < len(r.perturbation.pv_kwp) / n_customers < 0.65

    r = results[0]
    assert r.perturbation is not None
    fleet = PVOutputs(pv_units(r.perturbation, SPEC), SPEC, PVProfile())
    prepared = prepare_network(network, variant_id=r.variant_id, edits=r.script)
    loads = SyntheticProvider(customers_of(prepared), SyntheticProfiles()).draw(11, 0, 0, 0)
    pv = fleet.draw(11, 0, 0, 0, time_s=12 * 3600)
    ctx = prepared.ctx
    for c, p_w, q_var in zip(customers_of(prepared), loads.p_w, loads.q_var, strict=True):
        ctx.command(f"Edit {c.element} kW={float(p_w) / 1e3!r} kvar={float(q_var) / 1e3!r}")
    for u, p_w, q_var in zip(fleet.units, pv.p_w, pv.q_var, strict=True):
        ctx.command(f"Edit {u.generator} kW={float(p_w) / 1e3!r} kvar={float(q_var) / 1e3!r}")
    assert ctx.solve().converged
    assert setpoint_report(ctx).passes()  # every load and PV unit at its sampled setpoint
    r2 = residual(
        prepared.model, ctx.ynode_voltages(), injections(ctx, prepared.model), tolerance=1e-10
    )
    assert r2.passes()
    delivered = 0.0
    for u in fleet.units:
        ctx.api.Circuit.SetActiveElement(u.generator)
        delivered -= float(np.asarray(ctx.api.CktElement.Powers())[0::2].sum()) * 1e3
    assert delivered == pytest.approx(pv.p_w.sum(), rel=1e-9)
    assert pv.p_w.sum() > 0
