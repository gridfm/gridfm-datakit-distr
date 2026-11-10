"""The Kron view on catalog networks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from gridfm_datakit_distr.views.kron import KronView, kron_script

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]


def ybus_by_name(p: PreparedNetwork) -> dict[tuple[str, str], complex]:
    y = p.model.ybus().tocoo()
    names = [n.opendss_node for n in p.model.nodes]
    return {(names[r], names[c]): complex(v) for r, c, v in zip(y.row, y.col, y.data, strict=True)}


@pytest.mark.parametrize("network_id", ["csiro_enwl/n01_f1", "csiro_enwl/n21_f3"])
def test_the_view_equals_csiros_kron_files(
    network_master: Callable[[str], Path], network_id: str
) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    full = prepare_network(network)
    script = kron_script(full.ctx, full.model)
    assert script is not None
    view = prepare_network(network, variant_id=f"{network_id}/v0000", edits=script)
    theirs = prepare_network(network.model_copy(update={"master": network.representations["kron"]}))
    assert sorted(n.opendss_node for n in view.model.nodes) == sorted(
        n.opendss_node for n in theirs.model.nodes
    )
    ours, ref = ybus_by_name(view), ybus_by_name(theirs)
    scale = max(abs(v) for v in ref.values())
    worst = max(
        abs(ours.get(k, 0j) - ref.get(k, 0j))
        / max(abs(ours.get(k, 0j)), abs(ref.get(k, 0j)), 1e-9 * scale)
        for k in set(ours) | set(ref)
    )
    assert worst < 1e-6  # measured 2.0e-15 and 2.5e-15
    v_ref = {
        n.opendss_node: v
        for n, v in zip(theirs.model.nodes, theirs.ctx.ynode_voltages(), strict=True)
    }
    v = view.ctx.ynode_voltages()
    base = np.array([n.kv_base_ln for n in view.model.nodes]) * 1e3
    dv = max(abs(v[j] - v_ref[n.opendss_node]) / base[j] for j, n in enumerate(view.model.nodes))
    assert dv < 1e-9


@pytest.mark.parametrize("network_id", ["ieee13", "edp/industrial", "csiro_enwl/n01_f1"])
def test_the_view_per_scenario(network_master: Callable[[str], Path], network_id: str) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    cfg = GenerationConfig.model_validate(
        {
            "dataset_name": "t",
            "networks": {"include": [network_id]},
            "scenarios": {"n_scenarios_per_variant": 3},
            "settings": {"seed": 2},
        }
    )
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options))
    result = generate_variant(network, targets, cfg.variants, 2, 0, 0, options=options)
    runner = VariantRunner(network, result, cfg, 0)
    view = KronView(runner, network, result)
    assert view.identity == (network_id == "ieee13")
    for s in range(3):
        o = runner.run(s)
        assert o.pf_status == "converged", o.message
        k = view.solve(o, runner.registry.states.get(o.state_id or ""))
        assert k.status == "converged", k.message
        assert k.v is not None
        if view.identity:  # IEEE 13: the view is the full model (AC: within 1e-8 pu)
            assert o.v is not None
            assert np.array_equal(k.v, o.v)
