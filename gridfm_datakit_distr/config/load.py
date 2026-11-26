"""Read, validate and check a generation config.

Every problem is reported at once as ``<dotted.path>: <reason>`` lines in a :class:`ConfigError`.
Structure, types and ranges come from the pydantic models (``schema.py``); the rules here need
the network catalog or the capabilities of this version, and run once the structure is valid.
"""

from __future__ import annotations

import copy
import fnmatch
import importlib.util
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from gridfm_datakit_distr.config.schema import GenerationConfig, Networks
from gridfm_datakit_distr.model.ids import MAX_NETWORK_INDEX
from gridfm_datakit_distr.networks.catalog import Catalog, load_catalog
from gridfm_datakit_distr.scenarios.seeds import network_key

MAX_NETWORKS = MAX_NETWORK_INDEX + 1
"""Scenario ids hold 16 bits of network index."""

Issue = tuple[str, str]
"""(location, reason)."""


class ConfigError(ValueError):
    """An invalid config; :attr:`issues` lists every (location, reason)."""

    def __init__(self, source: str, issues: Sequence[Issue]) -> None:
        self.source = source
        self.issues = list(issues)
        lines = "\n".join(f"- {location}: {reason}" for location, reason in self.issues)
        super().__init__(f"invalid configuration {source}:\n{lines}")


@dataclass(frozen=True, slots=True)
class SelectedNetwork:
    """A catalog network of the dataset and the representation to load (None: its default)."""

    network_id: str
    representation: str | None


def load_config(path: Path, catalog: Catalog | None = None) -> GenerationConfig:
    """Read a YAML config file and validate it (:func:`validate_config`)."""
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(source, [("configuration", "no such file")]) from None
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
    except _DuplicateKey as exc:
        raise ConfigError(source, [(f"line {exc.line}", f"duplicate key {exc.key!r}")]) from None
    except yaml.YAMLError as exc:
        raise ConfigError(source, [("configuration", f"invalid YAML: {exc}")]) from None
    return validate_config(data, source, catalog)


def validate_config(
    data: Any, source: str = "<mapping>", catalog: Catalog | None = None
) -> GenerationConfig:
    """Validate a parsed config: the models first, then the rules of this module."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            source, [("configuration", f"expected a mapping, got {type(data).__name__}")]
        )
    try:
        config = GenerationConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            source, [_issue(error) for error in exc.errors(include_url=False)]
        ) from None
    catalog = catalog or load_catalog()
    issues = [issue for rule in _RULES for issue in rule(config)]
    selected, found = _selection(config.networks, catalog)
    ids = [s.network_id for s in selected]
    issues += found + _pv_locations(config, ids, catalog) + _trip_countries(config, ids, catalog)
    if issues:
        raise ConfigError(source, issues)
    return config


def stored_config(data: Mapping[str, Any]) -> GenerationConfig:
    """The resolved config of a dataset's manifest, without what was removed since it was
    written: the key ``labels.pf`` and the value ``as_is`` of ``normalisation.loads.model``
    (replaced by the default). Neither changes how the stored data is read."""
    data = copy.deepcopy(dict(data))
    data.get("labels", {}).pop("pf", None)
    loads = data.get("normalisation", {}).get("loads", {})
    if loads.get("model") == "as_is":
        del loads["model"]
    return GenerationConfig.model_validate(data)


def select_networks(networks: Networks, catalog: Catalog) -> list[SelectedNetwork]:
    """The networks of a valid config, ordered by id (network index of the seed tree)."""
    selected, issues = _selection(networks, catalog)
    if issues:
        raise ConfigError("networks", issues)
    return selected


def _selection(networks: Networks, catalog: Catalog) -> tuple[list[SelectedNetwork], list[Issue]]:
    issues: list[Issue] = []
    default: set[str] = set()
    named: dict[str, str] = {}
    for i, entry in enumerate(networks.include):
        location = f"networks.include[{i}]"
        pattern, _, representation = entry.partition("@")
        matched = sorted(n for n in catalog.networks if fnmatch.fnmatchcase(n, pattern))
        if not matched:
            issues.append((location, f"{pattern!r} matches no catalog network"))
            continue
        if not representation:
            default.update(matched)
            continue
        for network_id in matched:
            available = catalog.network(network_id).representations
            if representation not in available:
                names = ", ".join(sorted(available)) or "none"
                issues.append(
                    (
                        location,
                        f"{network_id} has no representation {representation!r} "
                        f"(available: {names})",
                    )
                )
            elif named.get(network_id, representation) != representation:
                issues.append(
                    (
                        location,
                        f"{network_id} is already selected with representation "
                        f"{named[network_id]!r}",
                    )
                )
            else:
                named[network_id] = representation
    selected = []
    for network_id in sorted(default | set(named)):
        chosen = named.get(network_id)
        if chosen is None and networks.neutral_representation == "kron":
            chosen = "kron" if "kron" in catalog.network(network_id).representations else None
        selected.append(SelectedNetwork(network_id, chosen))
    if len(selected) > MAX_NETWORKS:
        issues.append(("networks.include", f"{len(selected)} networks, at most {MAX_NETWORKS}"))
    keys: dict[int, str] = {}
    for s in selected:  # a shared key would correlate two networks' draws (p ~ n^2 / 2^65)
        other = keys.setdefault(network_key(s.network_id), s.network_id)
        if other != s.network_id:
            issues.append(("networks.include", f"{other} and {s.network_id} share a seed key"))
    return selected, issues


# ------------------------------------------------------------------ rules of this version


def _opf_requirements(config: GenerationConfig) -> list[Issue]:
    """OPF labels need CasADi (the ``opf`` extra)."""
    if not config.labels.opf.enabled or importlib.util.find_spec("casadi") is not None:
        return []
    return [
        (
            "labels.opf.enabled",
            "OPF labels need the 'opf' extra: pip install 'gridfm-datakit-distr[opf]'",
        )
    ]


_RULES: tuple[Callable[[GenerationConfig], list[Issue]], ...] = (_opf_requirements,)


def _pv_locations(config: GenerationConfig, selected: list[str], catalog: Catalog) -> list[Issue]:
    """PVGIS needs one location per selected network (none is built in), and every key of
    ``locations`` must match a catalog network."""
    pvgis = config.scenarios.pv_profile.pvgis
    where = "scenarios.pv_profile.pvgis.locations"
    issues = [
        (f"{where}.{key}", f"{key!r} matches no catalog network")
        for key in pvgis.locations
        if not any(fnmatch.fnmatchcase(n, key) for n in catalog.networks)
    ]
    if config.scenarios.pv_profile.provider != "pvgis":
        return issues
    keys = {n: pvgis.matching_keys(n) for n in selected}
    unplaced = [n for n, k in keys.items() if not k]
    if unplaced:
        issues.append((where, f"no location for {', '.join(unplaced)} (none is built in)"))
    issues += [
        (where, f"{n} is matched by several patterns: {', '.join(k)}")
        for n, k in keys.items()
        if len(k) > 1
    ]
    return issues


def _trip_countries(config: GenerationConfig, selected: list[str], catalog: Catalog) -> list[Issue]:
    """``der.pv.trip.mode: national`` needs national settings for every selected network's
    country (catalog ``country`` of its source)."""
    trip = config.der.pv.trip
    if not trip.enabled or trip.mode != "national":
        return []
    by_reason: dict[str, list[str]] = {}
    for n in selected:
        try:
            trip.settings(catalog.source(catalog.network(n).source).country)
        except ValueError as exc:
            by_reason.setdefault(str(exc), []).append(n)
    return [("der.pv.trip.mode", f"{', '.join(n)}: {r}") for r, n in by_reason.items()]


# ------------------------------------------------------------------ error formatting

_NO_INPUT = {"missing", "extra_forbidden", "value_error", "assertion_error"}
_NUMBER_TYPES = {"float_type", "float_parsing", "int_type", "int_parsing"}


def _location(loc: Sequence[int | str]) -> str:
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or "configuration"


def _issue(error: Mapping[str, Any]) -> Issue:
    message = str(error["msg"]).removeprefix("Value error, ")
    if error["type"] not in _NO_INPUT:
        value = error.get("input")
        text = repr(value)
        message += f", got {text if len(text) <= 60 else text[:57] + '...'}"
        if error["type"] in _NUMBER_TYPES and isinstance(value, str) and _yaml_float(value):
            message += f" (YAML reads it as text: write {_with_point(value)})"
    return _location(error["loc"]), message


_FLOAT = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)[eE][-+]?\d+$")


def _yaml_float(text: str) -> bool:
    """Whether ``text`` is an exponent float that YAML 1.1 (PyYAML) reads as a string."""
    return bool(_FLOAT.match(text)) and "." not in text.split("e")[0].split("E")[0]


def _with_point(text: str) -> str:
    mantissa, exponent = re.split("[eE]", text, maxsplit=1)
    return f"{mantissa}.0e{exponent}"


# ------------------------------------------------------------------ YAML loading


class _DuplicateKey(yaml.YAMLError):
    def __init__(self, key: object, line: int) -> None:
        super().__init__(f"duplicate key {key!r} at line {line}")
        self.key = key
        self.line = line


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that rejects duplicate mapping keys (PyYAML keeps the last one)."""


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise _DuplicateKey(key, key_node.start_mark.line + 1)
        seen.add(key)
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
