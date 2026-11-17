"""Tests for the variant generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import Variants
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.scenarios.seeds import network_key, variant_seed
from gridfm_datakit_distr.variants import generate
from gridfm_datakit_distr.variants.generate import (
    VariantError,
    generate_variant,
    generate_variants,
    variants_table,
)
from gridfm_datakit_distr.variants.perturbation import (
    check_edits,
    edit_script,
    sample_perturbation,
)
from gridfm_datakit_distr.variants.targets import VariantTargets, collect_targets
from tests.circuits import FOUR_WIRE, VARIANT_FEEDER, write_circuit

SEED = 7
PERTURBING: dict[str, Any] = {
    "n_variants": 3,
    "admittance": {
        "line_length_scale": {"distribution": "uniform", "low": 0.9, "high": 1.1},
        "neutral_impedance_scale": {"distribution": "uniform", "low": 0.5, "high": 2.0},
        "grounding_resistance_ohm": {"distribution": "choice", "values": [1.0, 5.0, 20.0]},
    },
    "phase_reallocation": {"enabled": True, "fraction": 0.5},
}


def variants(**changes: Any) -> Variants:
    return Variants.model_validate({**PERTURBING, **changes})


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Network, Path, PreparedNetwork]:
    root = tmp_path_factory.mktemp("variants")
    write_circuit(root / "src", VARIANT_FEEDER)
    network = Network(id="t/var", source="src", master="master.dss")
    return network, root, prepare_network(network, root=root)


@pytest.fixture(scope="module")
def targets(base: tuple[Network, Path, PreparedNetwork]) -> VariantTargets:
    return collect_targets(base[2])


# ------------------------------------------------------------------- targets


UNPERTURBED = Variants.model_validate(
    {
        "n_variants": 1,
        "admittance": {"line_length_scale": 1.0},
        "phase_reallocation": {"enabled": False},
    }
)
"""Variants as the defaults were before they perturbed: the source network."""


def test_targets(targets: VariantTargets) -> None:
    lines = {t.name: t.neutral for t in targets.lines}
    assert lines == {"a": (3,), "b": (3,), "lat": (1,)}  # the switch keeps its impedance
    assert {t.key for t in targets.groundings} == {"Reactor.grounding", "Transformer.t/w2"}
    loads = {t.name: (t.bus, t.phase, t.rest, t.candidates) for t in targets.loads}
    assert loads == {
        "h1": ("b1", "1", ("4",), ("2", "3")),
        "h2": ("b1", "2", ("4",), ("1", "3")),
        "h3": ("b2", "3", ("4",), ("1", "2")),
        "h4": ("b3", "1", ("4",), ("2", "3")),
    }  # h5 has a single phase at its bus, pp is phase to phase


def test_a_reactor_bonding_the_neutral_to_the_source_is_not_an_electrode(tmp_path: Path) -> None:
    # FOUR_WIRE grounds src.4 through a reactor and its Vsource has its star on node 0: the
    # reactor carries the whole neutral return current (CSIRO's pattern)
    write_circuit(tmp_path / "src", FOUR_WIRE)
    network = Network(id="t/four", source="src", master="master.dss")
    targets = collect_targets(prepare_network(network, root=tmp_path))
    assert targets.groundings == ()
    assert [t.element for t in targets.loads] == ["Load.house"]


# ------------------------------------------------------------------- sampling


def test_sampling_is_positional(targets: VariantTargets) -> None:
    config = variants()
    p = sample_perturbation(config, targets, SEED, 0, 0)
    assert p == sample_perturbation(config, targets, SEED, 0, 0)
    assert p != sample_perturbation(config, targets, SEED, 0, 1)
    assert p != sample_perturbation(config, targets, SEED, 0, 0, attempt=1)
    assert set(p.line_length) == {"Line.a", "Line.b", "Line.lat"}
    assert all(0.9 <= v < 1.1 for v in p.line_length.values())
    assert set(p.neutral_impedance) == {"Line.a", "Line.b", "Line.lat"}
    assert set(p.grounding_ohm) == {"Reactor.grounding", "Transformer.t/w2"}
    assert set(p.grounding_ohm.values()) <= {1.0, 5.0, 20.0}
    assert len(p.phases) == 2  # round(0.5 x 4 eligible loads)
    by_name = {t.element: t for t in targets.loads}
    for load, (old, new) in p.phases.items():
        t = by_name[load]
        assert old == f"{t.bus}.{t.phase}.4"
        bus, phase, neutral = new.split(".")
        assert (bus, neutral) == (t.bus, "4")
        assert phase in t.candidates


def test_quantities_draw_from_their_own_streams(targets: VariantTargets) -> None:
    on = sample_perturbation(variants(), targets, SEED, 0, 0)
    off = sample_perturbation(variants(phase_reallocation={"enabled": False}), targets, SEED, 0, 0)
    assert off.phases == {}
    assert (off.line_length, off.neutral_impedance) == (on.line_length, on.neutral_impedance)


def test_an_unperturbed_config_changes_nothing(base: tuple[Network, Path, PreparedNetwork]) -> None:
    network, root, prepared = base
    targets = collect_targets(prepared)
    p = sample_perturbation(UNPERTURBED, targets, SEED, 0, 0)
    assert edit_script(p, targets) == []
    result = generate_variant(network, targets, UNPERTURBED, SEED, 0, root=root)
    assert result.status == "ok"
    assert result.model is not None
    assert json.loads(result.record()["perturbation"]) == {"attempt": 0}
    for e, b in zip(result.model.elements, prepared.model.elements, strict=True):
        assert e.name == b.name
        if b.yprim is not None:
            assert e.yprim is not None
            assert np.array_equal(e.yprim, b.yprim)


# ------------------------------------------------------------------- realisation


def test_the_script_realises_the_perturbation(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    result = generate_variant(network, targets, variants(), SEED, 0, root=root)
    assert result.status == "ok", result.failures
    assert result.edit_deviation <= 1e-12
    p, model = result.perturbation, result.model
    assert p is not None
    assert model is not None
    matrix_edits = [c for c in result.script if "rmatrix=" in c]
    assert len(matrix_edits) == 3
    assert all(" cmatrix=" in c and " units=m length=" in c for c in matrix_edits)
    names = {e.name: e for e in model.elements}
    for load, (_, new) in p.phases.items():
        bus, phase, neutral = new.split(".")
        nodes = [
            model.nodes[k].opendss_node if k != GROUND else "0" for k in names[load].terminals[0]
        ]
        assert nodes == [f"{bus}.{phase}", f"{bus}.{neutral}"]


def test_the_edit_check_catches_the_units_trap(
    base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets
) -> None:
    network, root, _ = base
    p = sample_perturbation(variants(), targets, SEED, 0, 0)
    good = edit_script(p, targets)
    # the trap: matrices without cmatrix and units reset the line to "none" units
    trap = [c.split(" cmatrix=")[0] for c in good]
    for script, ok in ((good, True), (trap, False)):
        prepared = prepare_network(network, root=root, variant_id="t/var/v0000", edits=script)
        deviation = check_edits(p, targets, prepared.model)
        assert (deviation <= 1e-12) is ok, deviation


# ------------------------------------------------------------------- checks and records


def test_records(base: tuple[Network, Path, PreparedNetwork], targets: VariantTargets) -> None:
    network, root, _ = base
    result = generate_variant(network, targets, variants(), SEED, 5, root=root)
    assert result.model is not None
    row = result.record()
    assert row["variant_id"] == "t/var/v0005"
    assert row["network_id"] == "t/var"
    assert row["variant_idx"] == 5
    assert row["seed"] == variant_seed(SEED, network_key("t/var"), 5)
    assert (
        row["perturbation_script_sha1"]
        == hashlib.sha1("\n".join(result.script).encode("utf-8")).hexdigest()
    )
    assert row["n_nodes"] == len(result.model.nodes)
    assert row["n_elements"] == len(result.model.elements)
    sampled = json.loads(row["perturbation"])
    assert sampled["attempt"] == 0
    assert set(sampled) == {
        "attempt",
        "line_length_scale",
        "neutral_impedance_scale",
        "grounding_resistance_ohm",
        "phase_reallocation",
    }
    table = variants_table([result])
    schema.check_schema(table, schema.VARIANTS, "variants/variants")
    assert table.num_rows == 1


def test_failed_checks_are_resampled_then_recorded(
    base: tuple[Network, Path, PreparedNetwork],
    targets: VariantTargets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network, root, _ = base
    real = generate.physics_failures
    calls = {"n": 0}

    def flaky(prepared: PreparedNetwork) -> list[str]:
        calls["n"] += 1
        return ["C1: forced"] if calls["n"] <= 2 else real(prepared)

    monkeypatch.setattr(generate, "physics_failures", flaky)
    result = generate_variant(network, targets, variants(), SEED, 0, root=root)
    assert result.status == "ok"
    assert result.perturbation is not None
    assert result.perturbation.attempt == 2
    assert result.failures == ("attempt 0: C1: forced", "attempt 1: C1: forced")

    monkeypatch.setattr(generate, "physics_failures", lambda prepared: ["C2: forced"])
    failed = generate_variant(network, targets, variants(), SEED, 0, root=root)
    assert failed.status == "failed"
    assert failed.model is None
    assert len(failed.failures) == 3
    assert variants_table([failed]).num_rows == 0  # failures go to the manifest


def test_an_edit_that_misses_its_target_raises(
    base: tuple[Network, Path, PreparedNetwork],
    targets: VariantTargets,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network, root, _ = base
    monkeypatch.setattr(generate, "check_edits", lambda *args: 1e-3)
    with pytest.raises(VariantError, match="edits not realised"):
        generate_variant(network, targets, variants(), SEED, 0, root=root)


def test_generate_variants(base: tuple[Network, Path, PreparedNetwork]) -> None:
    network, root, _ = base
    results = generate_variants(network, variants(), SEED, root=root)
    assert [r.variant_id for r in results] == ["t/var/v0000", "t/var/v0001", "t/var/v0002"]
    assert all(r.status == "ok" for r in results)
    assert len({r.script_sha1 for r in results}) == 3
