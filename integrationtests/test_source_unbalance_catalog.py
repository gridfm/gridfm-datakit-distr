"""A negative-sequence source on catalog networks: a four-wire CSIRO feeder, three-wire IEEE 13."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.extract.source import read_source, scenario_e
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from tests.test_source_unbalance import READBACK_REL, UNBALANCE, sequences, thevenin

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]


def config(network_id: str, unbalance: object) -> GenerationConfig:
    return GenerationConfig.model_validate(
        {
            "dataset_name": "t",
            "networks": {"include": [network_id]},
            "scenarios": {"n_scenarios_per_variant": 3, "source_unbalance": unbalance},
            "settings": {"seed": 21, "residual_sample_rate": 1.0},
        }
    )


@pytest.mark.parametrize("network_id", ["csiro_enwl/n01_f1", "ieee13"])
def test_the_drawn_unbalance_is_the_catalog_source_s(
    network_master: Callable[[str], Path], network_id: str
) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    runners = []
    for unbalance in (0.0, UNBALANCE):
        cfg = config(network_id, unbalance)
        options = prepare_options(cfg)
        targets = collect_targets(prepare_network(network, options))
        result = generate_variant(
            network, targets, cfg.variants, 21, 0, options=options, der=cfg.der.pv
        )
        assert result.status == "ok", result.failures
        runners.append(VariantRunner(network, result, cfg, 0))
    balanced, r = runners
    assert r.model.nodes == balanced.model.nodes
    assert [e.name for e in r.model.elements] == [e.name for e in balanced.model.elements]
    source = read_source(r.ctx, r.model)
    bus = r.ctx.ynode_positions(r.model.nodes[int(n)].opendss_node for n in source.nodes)
    for s in range(3):
        o = r.run(s)
        assert o.pf_status == "converged", o.message
        assert o.residual_rel is not None
        assert o.residual_rel < 1e-6
        e = scenario_e(source.e_v, o.source_v_pu, o.source_u2, o.source_u2_angle_rad)
        got = thevenin(r.ctx, r.model, source)
        assert np.abs(got - e).max() <= READBACK_REL * np.abs(e).max()
        _, pos, neg = sequences(got)
        assert abs(neg / pos) == pytest.approx(o.source_u2, abs=READBACK_REL)
        # at the source bus the ratio moves by the drop across the source impedance (s58: up to
        # about 2e-4 on the CSIRO sources, 5e-5 on IEEE 13)
        _, pos, neg = sequences(r.ctx.ynode_voltages()[bus])
        assert abs(abs(neg / pos) - o.source_u2) <= 5e-4
