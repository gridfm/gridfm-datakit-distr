"""Variants of catalog networks: each passes C1 and C2, its edits are realised."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gridfm_datakit_distr.config.schema import Variants
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.variants.generate import EDIT_TOL, generate_variant, variants_table
from gridfm_datakit_distr.variants.targets import collect_targets

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]

# example perturbations, all drawn
EXAMPLE = Variants.model_validate(
    {
        "n_variants": 3,
        "admittance": {
            "line_length_scale": {"distribution": "uniform", "low": 0.9, "high": 1.1},
            "neutral_impedance_scale": {"distribution": "uniform", "low": 0.8, "high": 1.25},
            "grounding_resistance_ohm": {"distribution": "choice", "values": [1.0, 5.0, 20.0]},
        },
        "phase_reallocation": {"enabled": True, "fraction": 0.2},
    }
)


@pytest.mark.parametrize(
    ("network_id", "kinds"),
    [
        # four-wire LV, single-phase customers; its grounding reactor is the neutral's bond to
        # the source, so no earth electrode to perturb
        ("csiro_enwl/n01_f1", {"line_length_scale", "neutral_impedance_scale",
                               "phase_reallocation"}),
        ("csiro_enwl/n21_f3", {"line_length_scale", "neutral_impedance_scale",
                               "phase_reallocation"}),  # the median feeder, 83 A of return current
        # MV Kron and LV four-wire on km linecodes with capacitance, Rneut windings
        ("edp/industrial", {"line_length_scale", "neutral_impedance_scale",
                            "grounding_resistance_ohm"}),
        # 2165 movable single-phase loads, 68 Rneut windings
        ("edp/rural", {"line_length_scale", "neutral_impedance_scale",
                       "grounding_resistance_ohm", "phase_reallocation"}),
        # Kron model: lengths and phases only
        ("ieee123", {"line_length_scale", "phase_reallocation"}),
    ],
)  # fmt: skip
def test_variants_pass_c1_c2(
    network_master: Callable[[str], Path], network_id: str, kinds: set[str]
) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    targets = collect_targets(prepare_network(network))
    results = [generate_variant(network, targets, EXAMPLE, 11, k) for k in range(3)]
    assert len(results) == 3
    for r in results:
        assert r.status == "ok", r.failures
        assert r.edit_deviation <= EDIT_TOL
        assert r.perturbation is not None
        assert set(json.loads(r.perturbation.to_json())) - {"attempt"} == kinds
    table = variants_table(results)
    schema.check_schema(table, schema.VARIANTS, "variants/variants")
    assert table.column("variant_id").to_pylist() == [f"{network_id}/v{k:04d}" for k in range(3)]
