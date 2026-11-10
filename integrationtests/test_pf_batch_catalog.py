"""The per-scenario PF loop on catalog networks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner, prepare_options
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]


def config(network_id: str, **sections: Any) -> GenerationConfig:
    data: dict[str, Any] = {
        "dataset_name": "t",
        "networks": {"include": [network_id]},
        "variants": {
            "admittance": {
                "line_length_scale": {"distribution": "uniform", "low": 0.95, "high": 1.05}
            }
        },
        "scenarios": {"n_scenarios_per_variant": 6},
        "settings": {"seed": 13, "residual_sample_rate": 0.5},
        **sections,
    }
    return GenerationConfig.model_validate(data)


CASES = [
    ("ieee13", {}),  # regulators settled and jittered
    ("ieee34", {}),  # long feeder: as compiled at 0.70 pu, settled per scenario
    (
        "ieee_eulv",
        {
            "scenarios": {
                "n_scenarios_per_variant": 6,
                "profiles": {"provider": "network_loadshapes"},
            }
        },
    ),
    ("csiro_enwl/n01_f1", {"der": {"pv": {"penetration": 0.5}}}),
    pytest.param(
        "edp/rural",
        {
            "scenarios": {
                "n_scenarios_per_variant": 6,
                "profiles": {"provider": "edp_smart_meter"},
            },
            "der": {"pv": {"penetration": 0.3}},
        },
        marks=pytest.mark.slow,
    ),
]


def check_currents(runner: VariantRunner, o: ScenarioOutcome) -> None:
    """Extracted currents equal the engine's on 40 sampled elements (1e-8 relative)."""
    assert o.i_pd is not None
    assert o.t_extract_s > 0
    plan = runner.plan
    elements = list(runner.model.pd_elements())
    rng = np.random.default_rng(o.scenario_id)
    for j in rng.choice(len(elements), size=min(40, len(elements)), replace=False):
        e = elements[int(j)]
        runner.ctx.activate_element(e.name)
        raw = np.asarray(runner.ctx.api.CktElement.Currents())
        expected = raw[0::2] + 1j * raw[1::2]
        got = o.i_pd[plan.element_idx == e.idx]
        assert np.abs(got - expected).max() <= 1e-8 * max(np.abs(expected).max(), 1e-12), e.name


@pytest.mark.parametrize(("network_id", "sections"), CASES)
def test_scenarios_on_catalog_networks(
    network_master: Callable[[str], Path], network_id: str, sections: dict[str, Any]
) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    cfg = config(network_id, **sections)
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options))
    result = generate_variant(
        network, targets, cfg.variants, 13, 0, 0, options=options, der=cfg.der.pv
    )
    assert result.status == "ok", result.failures
    runner = VariantRunner(network, result, cfg, 0)
    outcomes = []
    for s in range(6):
        outcomes.append(runner.run(s))
        if outcomes[-1].pf_status == "converged":
            check_currents(runner, outcomes[-1])
    statuses = [o.pf_status for o in outcomes]
    # IEEE 34 with constant-power loads does not converge above ~1.2x its nominal load: 2 of these
    # 6 scenarios draw 1.28x and 1.66x and are recorded as not converged
    assert statuses.count("converged") >= 4, [(o.pf_status, o.message) for o in outcomes]
    for o in outcomes:
        assert o.pf_status != "error", o.message
        if o.pf_status == "converged":
            assert o.v is not None
            if o.residual_rel is not None:
                assert o.residual_rel < 1e-6
    # bitwise order independence on a fresh runner, backwards
    again = VariantRunner(network, result, cfg, 0)
    for s in reversed(range(6)):
        o, a = outcomes[s], again.run(s)
        assert (a.pf_status, a.state_id) == (o.pf_status, o.state_id)
        if o.v is not None:
            assert a.v is not None
            assert np.array_equal(a.v, o.v)
