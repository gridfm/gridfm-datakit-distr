"""Tests for the generation config: models, loader, rules, reference doc.

Error messages are matched as ``<dotted.path>: <reason>`` lines, as gridfm-datakit's config tests
do.
"""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config import (
    ConfigError,
    GenerationConfig,
    SelectedNetwork,
    load_config,
    select_networks,
    validate_config,
)
from gridfm_datakit_distr.config.reference import reference_markdown
from gridfm_datakit_distr.config.value_spec import Fixed, Uniform
from gridfm_datakit_distr.networks.catalog import load_catalog

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "scripts" / "config").glob("*.yaml"))

MINIMAL: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["ieee13"]},
    "scenarios": {"n_scenarios_per_variant": 10},
    "settings": {"seed": 1},
}


def with_(path: str, value: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """A copy of ``base`` (default MINIMAL) with ``value`` set at the dotted ``path``."""
    data = copy.deepcopy(MINIMAL if base is None else base)
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value
    return data


def issues(data: Any) -> list[str]:
    with pytest.raises(ConfigError) as info:
        validate_config(data)
    lines = [f"{loc}: {message}" for loc, message in info.value.issues]
    assert str(info.value).splitlines()[1:] == [f"- {line}" for line in lines]
    return lines


def one_issue(data: Any, path: str, pattern: str) -> None:
    found = issues(data)
    assert len(found) == 1, found
    assert re.match(f"^{re.escape(path)}: {pattern}", found[0]), found[0]


# ------------------------------------------------------------------- valid configs


def test_a_minimal_config_takes_the_documented_defaults() -> None:
    config = validate_config(MINIMAL)
    assert isinstance(config, GenerationConfig)
    assert config.networks.neutral_representation == "explicit"
    assert (config.networks.voltage_limits.vmin_pu, config.networks.voltage_limits.vmax_pu) == (
        0.94,
        1.10,
    )
    assert config.normalisation.loads.model == "constant_power"
    assert config.normalisation.s_base_va == "auto"
    # defaults perturb the variants (four per network)
    assert config.variants.n_variants == 4
    assert config.variants.admittance.line_length_scale == Uniform(0.9, 1.1)
    assert config.variants.admittance.grounding_resistance_ohm is None
    assert config.variants.phase_reallocation.enabled
    assert config.der.pv.penetration == Uniform(0.0, 0.8)
    assert config.scenarios.profiles.synthetic.global_scale == Uniform(0.2, 1.2)
    assert config.scenarios.source_voltage_pu == Uniform(0.98, 1.05)
    assert config.settings.num_processes is None  # all cores
    assert config.settings.validation_tolerance == 1e-10


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_example_configs_validate(path: Path) -> None:
    config = load_config(path)
    assert config.dataset_name


def test_there_are_example_configs() -> None:
    assert {p.name for p in CONFIGS} >= {
        "ieee13_small.yaml",
        "csiro_enwl_pf.yaml",
        "edp_rural.yaml",
    }


def test_dump_round_trip() -> None:
    config = load_config(REPO / "scripts" / "config" / "csiro_enwl_pf.yaml")
    dumped = config.model_dump(mode="json")
    assert validate_config(dumped) == config
    assert dumped["variants"]["admittance"]["line_length_scale"] == {
        "distribution": "uniform",
        "low": 0.9,
        "high": 1.1,
    }


def test_specs_take_numbers_and_distributions() -> None:
    config = validate_config(with_("der.pv.kwp", 4))
    assert config.der.pv.kwp == Fixed(4.0)
    spec = {"distribution": "normal", "mean": 1.0, "std": 0.02}
    assert validate_config(with_("scenarios.source_voltage_pu", spec)).scenarios.source_voltage_pu


def test_voltage_limit_preset() -> None:
    limits = validate_config(with_("networks.voltage_limits", {"preset": "ansi_a"}))
    assert (limits.networks.voltage_limits.vmin_pu, limits.networks.voltage_limits.vmax_pu) == (
        0.95,
        1.05,
    )
    same = with_("networks.voltage_limits", {"preset": "uk_esqcr", "vmin_pu": 0.94})
    assert validate_config(same).networks.voltage_limits.vmax_pu == 1.10


# ------------------------------------------------------------------- structure errors


def test_unknown_keys_are_rejected_at_their_path() -> None:
    one_issue(
        with_("variants.admittance.line_lenght_scale", 1.0),
        "variants.admittance.line_lenght_scale",
        "Extra inputs are not permitted",
    )
    one_issue(with_("sedd", 1), "sedd", "Extra inputs are not permitted")


@pytest.mark.parametrize(
    ("drop", "path"),
    [
        ("dataset_name", "dataset_name"),
        ("networks", "networks"),
        ("scenarios", "scenarios"),
        ("settings", "settings"),
    ],
)
def test_required_sections(drop: str, path: str) -> None:
    data = copy.deepcopy(MINIMAL)
    del data[drop]
    one_issue(data, path, "Field required")


def test_required_fields_inside_sections() -> None:
    one_issue(with_("settings", {}), "settings.seed", "Field required")
    one_issue(with_("networks", {}), "networks.include", "Field required")


def test_types_are_not_coerced() -> None:
    one_issue(with_("settings.seed", "1"), "settings.seed", "Input should be a valid integer")
    one_issue(
        with_("settings.overwrite", "yes"), "settings.overwrite", "Input should be a valid boolean"
    )
    one_issue(with_("views.kron", 1), "views.kron", "Input should be a valid boolean")


def test_yaml_text_numbers_get_a_hint() -> None:
    # PyYAML (YAML 1.1) reads 1e-10 without a decimal point as text
    one_issue(
        with_("settings.validation_tolerance", "1e-10"),
        "settings.validation_tolerance",
        r"Input should be a valid number, got '1e-10' \(YAML reads it as text: write 1\.0e-10\)",
    )


def test_list_items_are_located_by_index() -> None:
    one_issue(
        with_("networks.include", ["ieee13", ""]),
        "networks.include[1]",
        "String should have at least 1 character",
    )


def test_ranges() -> None:
    one_issue(
        with_("settings.chunk_size", 0),
        "settings.chunk_size",
        "Input should be greater than or equal to 1",
    )
    one_issue(
        with_("settings.residual_sample_rate", 1.5),
        "settings.residual_sample_rate",
        "Input should be less than or equal to 1",
    )
    one_issue(
        with_("variants.n_variants", 10_001),
        "variants.n_variants",
        "Input should be less than or equal to 10000",
    )
    one_issue(
        with_("normalisation.loads.vminpu", 1.0),
        "normalisation.loads.vminpu",
        "Input should be less than 1",
    )


def test_literals_list_their_values() -> None:
    one_issue(
        with_("networks.neutral_representation", "four_wire"),
        "networks.neutral_representation",
        "Input should be 'explicit', 'kron' or 'as_is', got 'four_wire'",
    )


def test_dataset_name_is_a_directory_name() -> None:
    one_issue(with_("dataset_name", "my data"), "dataset_name", "String should match pattern")


def test_s_base() -> None:
    assert validate_config(with_("normalisation.s_base_va", 1e6)).normalisation.s_base_va == 1e6
    one_issue(
        with_("normalisation.s_base_va", -5),
        "normalisation.s_base_va",
        r"must be 'auto' or a number > 0 \(VA\), got -5",
    )


def test_validation_tolerance_respects_the_setpoint_check() -> None:
    one_issue(
        with_("settings.validation_tolerance", 1.0e-6),
        "settings.validation_tolerance",
        r"Input should be less than or equal to 0\.00000001, got 1e-06",
    )


def test_several_errors_are_reported_together() -> None:
    data = with_("settings.chunk_size", 0, with_("views.kron", "no"))
    found = issues(data)
    assert [line.split(":")[0] for line in found] == ["views.kron", "settings.chunk_size"]


# ------------------------------------------------------------------- value specs


def test_spec_errors_are_located() -> None:
    one_issue(
        with_(
            "variants.admittance.line_length_scale",
            {"distribution": "uniform", "low": 1.1, "high": 0.9},
        ),
        "variants.admittance.line_length_scale",
        "low must be below high, got 1.1 and 0.9",
    )
    one_issue(
        with_("der.pv.kwp", {"distribution": "uniform", "low": 0, "high": 6}),
        "der.pv.kwp",
        r"every value must be > 0.0, got 0.0",
    )


def test_spec_ceilings() -> None:
    one_issue(
        with_("der.pv.penetration", {"distribution": "uniform", "low": 0, "high": 1.5}),
        "der.pv.penetration",
        "every value must be <= 1, but draws reach 1.5",
    )
    # an unbounded normal reaches mean + 6 std: a ceiling needs an explicit max
    one_issue(
        with_(
            "scenarios.profiles.synthetic.power_factor",
            {"distribution": "normal", "mean": 0.95, "std": 0.01},
        ),
        "scenarios.profiles.synthetic.power_factor",
        "every value must be <= 1, but draws reach 1.01",
    )
    bounded = {"distribution": "normal", "mean": 0.95, "std": 0.01, "max": 1.0}
    assert validate_config(with_("scenarios.profiles.synthetic.power_factor", bounded))


# ------------------------------------------------------------------- cross-field rules


def test_voltage_limits() -> None:
    one_issue(
        with_("networks.voltage_limits", {"vmin_pu": 1.05, "vmax_pu": 0.95}),
        "networks.voltage_limits",
        "vmin_pu must be below vmax_pu, got 1.05 and 0.95",
    )
    one_issue(
        with_("networks.voltage_limits", {"preset": "ansi_a", "vmin_pu": 0.9}),
        "networks.voltage_limits",
        "preset 'ansi_a' sets vmin_pu 0.95 and vmax_pu 1.05; got vmin_pu 0.9",
    )


def test_synthetic_settings_need_the_synthetic_provider() -> None:
    data = with_(
        "scenarios.profiles", {"provider": "network_loadshapes", "synthetic": {"local_sigma": 0.1}}
    )
    one_issue(data, "scenarios.profiles", "synthetic is used only with provider 'synthetic'")


def test_something_must_be_labelled() -> None:
    one_issue(with_("labels", {"pf": False}), "labels", "nothing to label")


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("variants.reconfiguration.enabled", True, "switch reconfiguration is not in v0.1"),
        ("baselines.lindist_pf", True, "LinDist3Flow baselines come with v1.0"),
        ("baselines.lindist_opf", True, "LinDist3Flow baselines come with v1.0"),
        ("measurements.enabled", True, "measurement configurations come with v1.0"),
        ("scenarios.profiles.provider", "fluvius", "the Fluvius profiles wait for their licence"),
    ],
)
def test_features_not_in_this_version(path: str, value: Any, reason: str) -> None:
    one_issue(with_(path, value), path, reason)


def test_networks_must_have_distinct_seed_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from gridfm_datakit_distr.config import load

    monkeypatch.setattr(load, "network_key", lambda network_id: 7)  # force a collision
    one_issue(
        with_("networks.include", ["ieee13", "ieee34"]),
        "networks.include",
        "ieee13 and ieee34 share a seed key",
    )


def test_opf_labels_are_accepted() -> None:
    pytest.importorskip("casadi")
    config = validate_config(with_("labels.opf.enabled", True))
    assert (config.labels.opf.costs, config.labels.opf.time_limit_s) == ("economic", 300.0)


def test_opf_labels_need_constant_power_loads() -> None:
    pytest.importorskip("casadi")
    data = with_("normalisation.loads.model", "as_is", with_("labels.opf.enabled", True))
    one_issue(data, "labels.opf.enabled", "OPF labels need normalisation.loads.model")


def test_opf_labels_need_the_opf_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "casadi" else real(name, *a)
    )
    one_issue(with_("labels.opf.enabled", True), "labels.opf.enabled", "OPF labels need the 'opf'")


def test_the_cost_presets_are_economic_and_sampled() -> None:
    pytest.importorskip("casadi")
    sampled = validate_config(with_("labels.opf", {"enabled": True, "costs": "sampled"}))
    assert sampled.labels.opf.sampled.source_c1.sample(np.random.default_rng(0)) == 1.0
    found = issues(with_("labels.opf", {"enabled": True, "costs": "shared"}))
    assert found[0].startswith("labels.opf.costs: Input should be 'economic' or 'sampled'"), found


# ------------------------------------------------------------------- network selection


def test_patterns_must_match_catalog_networks() -> None:
    one_issue(
        with_("networks.include", ["ieee13", "ieee14"]),
        "networks.include[1]",
        "'ieee14' matches no catalog network",
    )


def test_representations() -> None:
    catalog = load_catalog()
    config = validate_config(
        with_("networks.include", ["ieee8500@unbalanced", "csiro_enwl/n01_f*"])
    )
    selected = select_networks(config.networks, catalog)
    assert selected[0] == SelectedNetwork("csiro_enwl/n01_f1", None)
    assert selected[-1] == SelectedNetwork("ieee8500", "unbalanced")
    assert [s.network_id for s in selected] == sorted(s.network_id for s in selected)
    one_issue(
        with_("networks.include", ["ieee13@kron"]),
        "networks.include[0]",
        "ieee13 has no representation 'kron'",
    )


def test_kron_selects_the_sources_kron_files_where_they_exist() -> None:
    catalog = load_catalog()
    data = with_(
        "networks", {"include": ["csiro_enwl/n01_f1", "ieee13"], "neutral_representation": "kron"}
    )
    selected = select_networks(validate_config(data).networks, catalog)
    assert selected == [
        SelectedNetwork("csiro_enwl/n01_f1", "kron"),
        SelectedNetwork("ieee13", None),
    ]


def test_a_suffix_overrides_a_pattern_and_two_suffixes_conflict() -> None:
    catalog = load_catalog()
    data = with_("networks.include", ["csiro_enwl/n01_f1*", "csiro_enwl/n01_f1@kron"])
    assert select_networks(validate_config(data).networks, catalog) == [
        SelectedNetwork("csiro_enwl/n01_f1", "kron")
    ]
    one_issue(
        with_("networks.include", ["csiro_enwl/n01_f1@kron", "csiro_enwl/n01_f1@phase_to_neutral"]),
        "networks.include[1]",
        "csiro_enwl/n01_f1 is already selected with representation 'kron'",
    )


# ------------------------------------------------------------------- YAML loading


def test_yaml_errors(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("dataset_name: a\ndataset_name: b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"line 2: duplicate key 'dataset_name'"):
        load_config(path)
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"configuration: expected a mapping, got list"):
        load_config(path)
    path.write_text("dataset_name: [a\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"YAML"):
        load_config(path)
    with pytest.raises(ConfigError, match=r"no such file"):
        load_config(tmp_path / "missing.yaml")


def test_errors_name_the_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("dataset_name: a\n", encoding="utf-8")
    with pytest.raises(
        ConfigError, match=rf"^invalid configuration {re.escape(str(path))}:\n- networks: "
    ):
        load_config(path)


# ------------------------------------------------------------------- reference doc


def test_reference_doc_is_current() -> None:
    expected = reference_markdown()
    current = (REPO / "docs" / "manual" / "config.md").read_text(encoding="utf-8")
    assert current == expected, "regenerate: uv run python -m gridfm_datakit_distr.config.reference"


def test_every_field_is_documented() -> None:
    text = reference_markdown()
    assert "| `settings.seed` | integer ≥ 0 | required |" in text
    expected = "`{distribution: uniform, low: 0.0, high: 0.8}`"
    assert f"| `der.pv.penetration` | value spec, ≥ 0 and ≤ 1 | {expected} |" in text
    assert "TODO" not in text


def test_generate_notes_networks_without_an_explicit_neutral(tmp_path: Path) -> None:
    """``explicit`` (the default) loads each network's own model: a Kron-reduced one is said so."""
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app

    config = tmp_path / "c.yaml"
    config.write_text(
        "dataset_name: d\nscenarios: {n_scenarios_per_variant: 1}\nsettings: {seed: 1}\n"
        "networks:\n  include: [ieee13, edp/rural]\n"
    )
    result = CliRunner().invoke(app, ["generate", str(config), "--cache", str(tmp_path / "none")])
    assert "note: no explicit neutral in ieee13 (Kron-reduced source model)" in result.output
    assert "edp/rural" not in result.output.split("note:")[1].splitlines()[0]
    explicit = config.read_text() + "  neutral_representation: as_is\n"
    config.write_text(explicit)
    result = CliRunner().invoke(app, ["generate", str(config), "--cache", str(tmp_path / "none")])
    assert "note:" not in result.output
