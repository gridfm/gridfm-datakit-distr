"""`scripts/config/default.yaml` shows the defaults: every value it sets equals the schema's."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from gridfm_datakit_distr.config.load import load_config
from gridfm_datakit_distr.config.schema import GenerationConfig

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "scripts" / "config" / "default.yaml"
REQUIRED = (("dataset_name",), ("networks", "include"), ("scenarios", "n_scenarios_per_variant"),
            ("settings", "seed"))  # fmt: skip
"""Keys without a default: the file must set them."""


def leaves(data: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if isinstance(data, dict) and "distribution" not in data:  # a value spec is one value
        for key, value in data.items():
            yield from leaves(value, (*path, key))
    else:
        yield path


def at(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        data = data[key]
    return data


def test_every_value_of_default_yaml_is_the_schema_default() -> None:
    raw = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))
    given = load_config(DEFAULT_YAML).model_dump(mode="json")
    minimal: dict[str, Any] = {}
    for path in REQUIRED:
        node = minimal
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = at(raw, path)
    default = GenerationConfig.model_validate(minimal).model_dump(mode="json")
    differ = [
        ".".join(p) for p in leaves(raw) if p not in REQUIRED and at(given, p) != at(default, p)
    ]
    assert not differ, f"default.yaml differs from the schema defaults at: {differ}"
