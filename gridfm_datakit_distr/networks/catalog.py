"""Network catalog: sources (what to download and how to verify it) and networks (masters).

The catalog lives in ``catalog.yaml`` next to this module; per-source checksums live in
``manifests/<source>.sha256`` (``sha256sum`` format).
"""

from __future__ import annotations

import functools
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

NeutralRepresentation = Literal["explicit", "kron", "mixed"]

_PACKAGE = "gridfm_datakit_distr.networks"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CatalogError(ValueError):
    """The catalog or a manifest is malformed or inconsistent."""


def safe_relative_path(path: str) -> str:
    """Validate a POSIX path relative to a source directory (no absolute paths, no '..')."""
    pure = PurePosixPath(path)
    if not path or "\\" in path or pure.is_absolute() or ".." in pure.parts:
        raise CatalogError(f"unsafe relative path: {path!r}")
    return str(pure)


def parse_manifest(text: str, origin: str = "<manifest>") -> dict[str, str]:
    """Parse ``<sha256>  <relative path>`` lines (``sha256sum`` format); '#' starts a comment."""
    entries: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        digest, sep, path = line.partition("  ")
        if not sep or not _SHA256.match(digest):
            raise CatalogError(f"{origin}:{lineno}: expected '<sha256>  <path>', got {line!r}")
        path = safe_relative_path(path)
        if path in entries:
            raise CatalogError(f"{origin}:{lineno}: duplicate entry {path!r}")
        entries[path] = digest
    return entries


def format_manifest(entries: Mapping[str, str]) -> str:
    """Canonical manifest text: one ``<sha256>  <path>`` line per file, sorted by path."""
    return "".join(f"{entries[path]}  {path}\n" for path in sorted(entries))


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _SourceBase(_Model):
    id: str
    title: str
    homepage: str
    license: str
    citation: str
    doi: str | None = None
    notes: str = ""
    manifest: dict[str, str] = Field(default_factory=dict)
    """Relative path -> SHA-256 of every file of the source (from ``manifests/<id>.sha256``)."""

    @property
    def manifest_sha256(self) -> str:
        """Fingerprint of the manifest; a cached copy is valid only for the same fingerprint."""
        return hashlib.sha256(format_manifest(self.manifest).encode()).hexdigest()


class GithubSource(_SourceBase):
    """Files of a GitHub repository at a pinned commit (verified against git blob SHA-1 too)."""

    kind: Literal["github"]
    repo: str
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    subdir: str
    include: list[str]
    """Paths relative to ``subdir``; entries ending with '/' select whole folders."""


class CsiroDapSource(_SourceBase):
    """A CSIRO Data Access Portal collection version, listed through the DAP web service."""

    kind: Literal["csiro_dap"]
    collection: int
    version: int
    api_base: str = "https://data.csiro.au/dap/ws/v2"
    exclude: list[str] = Field(default_factory=list)
    """fnmatch patterns of listed files to skip (e.g. macOS ``.DS_Store``)."""


class ArchivesSource(_SourceBase):
    """Archives downloaded from fixed URLs and extracted into ``<archive stem>/``."""

    kind: Literal["archives"]
    archives: dict[str, str]
    """Archive file name -> download URL."""


Source = Annotated[GithubSource | CsiroDapSource | ArchivesSource, Field(discriminator="kind")]
SourceT = GithubSource | CsiroDapSource | ArchivesSource


class Network(_Model):
    """A network: a master file inside a source."""

    id: str
    source: str
    master: str
    """Path of the master file relative to the source directory."""
    neutral: NeutralRepresentation | None = None
    """Expected neutral classification of ``master`` (None: not yet known)."""
    representations: dict[str, str] = Field(default_factory=dict)
    """Alternative models of the same feeder: representation name -> master path."""
    extra_commands: list[str] = Field(default_factory=list)
    """OpenDSS commands run right after the master (``DssContext.load``): documented
    corrections of the source model, each justified in a comment of ``catalog.yaml``."""
    master_stop_before: str | None = None
    """Compile only the lines of ``master`` before this line (compared stripped, ignoring case):
    for masters that go on to run simulations and exports (EDP)."""
    base_loads: Literal["as_is", "loadshape_coincident_peak"] = "as_is"
    """``loadshape_coincident_peak``: each load's base kW and kvar become ``c`` times the mean of
    its daily (else yearly) loadshape, ``c`` the network's coincident peak over the profile window
    divided by the sum of the means (``normalize.loads.loads_at_coincident_peak``), for sources
    whose base values are placeholders (EDP: kW = kvar = 1). Base kW at the peak is OpenDSS's
    convention, as in the other sources."""
    nominal_kv_ln: dict[float, float] = Field(default_factory=dict)
    """Nominal phase-to-neutral voltage (kV) of the buses of a voltage level, keyed by the level's
    line-to-line base (kV), where it differs from the base (e.g. a 230 V network on a 416 V base);
    voltage limits are per unit of it. Levels not listed: the nominal voltage is the base."""
    load_level: bool = True
    """Whether the network's variants draw a load level ``L`` with synthetic loads
    (``scenarios.profiles.synthetic.variant_scale``); false: ``L = 1``, nothing drawn. False for
    measured nominal loads (EDP) and for MV feeders, whose loads are spot or aggregate loads over
    many substation areas while ``L`` is measured on LV feeders of one area each (IEEE MV)."""


class Collection(_Model):
    """A rule that expands the master files of a source manifest into networks."""

    source: str
    id: str
    """``str.format`` template over the named groups of ``match`` (numeric groups as int)."""
    match: str
    """Regex over manifest paths with a ``representation`` group and the id groups."""
    representations: dict[str, str]
    """Folder (value of the ``representation`` group) -> representation name."""
    default: str
    neutral: NeutralRepresentation | None = None
    nominal_kv_ln: dict[float, float] = Field(default_factory=dict)
    """Nominal phase-to-neutral voltage (kV) of the buses of a voltage level, keyed by the level's
    line-to-line base (kV), where it differs from the base (e.g. a 230 V network on a 416 V base);
    voltage limits are per unit of it. Levels not listed: the nominal voltage is the base."""


class _CatalogFile(_Model):
    sources: dict[str, Source]
    networks: list[Network] = Field(default_factory=list)
    collections: list[Collection] = Field(default_factory=list)


@dataclass(frozen=True)
class Catalog:
    """Sources and networks, validated and with collections expanded."""

    sources: dict[str, SourceT]
    networks: dict[str, Network]

    def source(self, source_id: str) -> SourceT:
        """The source with id ``source_id``."""
        try:
            return self.sources[source_id]
        except KeyError:
            raise KeyError(f"unknown source {source_id!r}; known: {sorted(self.sources)}") from None

    def network(self, network_id: str) -> Network:
        """The network with id ``network_id``."""
        try:
            return self.networks[network_id]
        except KeyError:
            raise KeyError(f"unknown network {network_id!r}") from None

    def source_for(self, ident: str) -> SourceT:
        """Resolve a source id or a network id to its source."""
        if ident in self.sources:
            return self.sources[ident]
        if ident in self.networks:
            return self.sources[self.networks[ident].source]
        raise KeyError(
            f"{ident!r} is neither a source nor a network; sources: {sorted(self.sources)}; "
            f"networks e.g. {sorted(self.networks)[:6]}"
        )


def _expand(collection: Collection, source: SourceT) -> list[Network]:
    pattern = re.compile(collection.match)
    feeders: dict[tuple[tuple[str, Any], ...], dict[str, str]] = {}
    for path in sorted(source.manifest):
        found = pattern.match(path)
        if found is None:
            continue
        groups = found.groupdict()
        folder = groups.pop("representation")
        if folder not in collection.representations:
            raise CatalogError(f"{path}: unknown representation folder {folder!r}")
        key = tuple(sorted((k, int(v) if v.isdigit() else v) for k, v in groups.items()))
        masters = feeders.setdefault(key, {})
        name = collection.representations[folder]
        if name in masters:
            raise CatalogError(f"{path}: two '{name}' masters for {dict(key)}")
        masters[name] = path
    networks = []
    for key, masters in sorted(feeders.items()):
        if collection.default not in masters:
            raise CatalogError(f"{dict(key)}: no '{collection.default}' master in {source.id}")
        networks.append(
            Network(
                id=collection.id.format(**dict(key)),
                source=collection.source,
                master=masters[collection.default],
                neutral=collection.neutral,
                representations=masters,
                nominal_kv_ln=collection.nominal_kv_ln,
            )
        )
    return networks


def _read_manifest(source_id: str) -> dict[str, str]:
    path = resources.files(_PACKAGE) / "manifests" / f"{source_id}.sha256"
    if not path.is_file():
        return {}
    return parse_manifest(path.read_text(encoding="utf-8"), origin=f"manifests/{source_id}.sha256")


def build_catalog(raw: Mapping[str, Any], manifests: Mapping[str, dict[str, str]]) -> Catalog:
    """Validate a parsed catalog document against the given manifests."""
    data = dict(raw)
    data["sources"] = {
        key: {**value, "id": key, "manifest": manifests.get(key, {})}
        for key, value in (raw.get("sources") or {}).items()
    }
    try:
        parsed = _CatalogFile.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(str(exc)) from exc

    sources: dict[str, SourceT] = dict(parsed.sources)
    networks: dict[str, Network] = {}
    expanded = [n for c in parsed.collections for n in _expand(c, sources[c.source])]
    for network in [*parsed.networks, *expanded]:
        if network.id in networks:
            raise CatalogError(f"duplicate network id {network.id!r}")
        if network.source not in sources:
            raise CatalogError(f"{network.id}: unknown source {network.source!r}")
        source = sources[network.source]
        for path in {network.master, *network.representations.values()}:
            safe_relative_path(path)
            # Masters of file sources must be manifest files (archives are extracted later).
            if source.kind != "archives" and source.manifest and path not in source.manifest:
                raise CatalogError(f"{network.id}: master {path!r} is not in the manifest")
        networks[network.id] = network
    return Catalog(sources=sources, networks=networks)


@functools.cache
def load_catalog() -> Catalog:
    """The packaged catalog (``catalog.yaml`` + ``manifests/``)."""
    text = resources.files(_PACKAGE).joinpath("catalog.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    manifests = {key: _read_manifest(key) for key in (raw.get("sources") or {})}
    return build_catalog(raw, manifests)
