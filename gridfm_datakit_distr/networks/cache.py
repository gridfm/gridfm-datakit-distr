"""Location and state of the network cache (outside the repository, never committed)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gridfm_datakit_distr.networks.catalog import Network, SourceT

CACHE_ENV = "GRIDFM_DISTR_CACHE"
MARKER = ".fetch.json"
"""File written in a source directory once all its files are verified."""
INSTALLED = ".fetch.sha256"
"""SHA-256 of every file installed in a source directory (``sha256sum`` format), checked by
``fetch.verify_cached`` before each use."""


def cache_root() -> Path:
    """``$GRIDFM_DISTR_CACHE`` if set, else ``~/.cache/gridfm_datakit_distr``."""
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "gridfm_datakit_distr"


def source_dir(source_id: str, root: Path | None = None) -> Path:
    """Directory of a fetched source."""
    return (root if root is not None else cache_root()) / source_id


def read_marker(source_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """The fetch marker of a source, or None if the source was never fetched."""
    path = source_dir(source_id, root) / MARKER
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def is_fetched(source: SourceT, root: Path | None = None) -> bool:
    """True if the cached copy was verified against the current manifest."""
    marker = read_marker(source.id, root)
    return marker is not None and marker.get("manifest_sha256") == source.manifest_sha256


def master_path(network: Network, root: Path | None = None) -> Path:
    """Absolute path of a network's master file in the cache."""
    return source_dir(network.source, root) / network.master
