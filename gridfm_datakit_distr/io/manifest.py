"""The dataset manifest: resolved config, its hash, versions, networks
and the work units with their status. Written atomically after every change.

``config_hash`` covers what determines the data: the resolved config without the execution
settings (``data_dir``, ``num_processes``, ``contexts_per_process``, ``overwrite``). A run with
another hash is a different dataset (resume needs the same hash).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from gridfm_datakit_distr.io.schema import SCHEMA_VERSION

UnitStatus = Literal["pending", "done", "failed"]
MANIFEST = "manifest.json"
EXECUTION_ONLY = ("data_dir", "num_processes", "contexts_per_process", "overwrite")


def config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the resolved config without its execution-only settings."""
    data = json.loads(json.dumps(config))
    for key in EXECUTION_ONLY:
        data.get("settings", {}).pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file next to ``path`` and rename it over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(slots=True)
class Unit:
    """A chunk of scenarios of one variant: ``scenarios [first, stop)``."""

    network_id: str
    network_idx: int
    variant_idx: int
    chunk: int
    first: int
    stop: int
    status: UnitStatus = "pending"
    message: str | None = None
    peak_rss_mib: float | None = None
    """Peak resident memory of the process that ran the unit, up to its end (MiB)."""

    @property
    def key(self) -> str:
        return f"{self.network_idx:05d}-{self.variant_idx:04d}-{self.chunk:05d}"


@dataclass(slots=True)
class Manifest:
    root: Path
    config: dict[str, Any]
    config_hash: str
    versions: dict[str, str]
    networks: list[dict[str, Any]] = field(default_factory=list)
    """``{"network_id", "network_idx", "representation"}`` in network order."""
    units: dict[str, Unit] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def create(cls, root: Path, config: dict[str, Any], versions: dict[str, str]) -> Manifest:
        manifest = cls(root, config, config_hash(config), versions)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, root: Path) -> Manifest:
        data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        return cls(
            root=root,
            config=data["config"],
            config_hash=data["config_hash"],
            versions=data["versions"],
            networks=data["networks"],
            units={k: Unit(**u) for k, u in data["units"].items()},
            schema_version=data["schema_version"],
        )

    def save(self) -> None:
        data = {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "versions": self.versions,
            "config": self.config,
            "networks": self.networks,
            "units": {k: asdict(u) for k, u in sorted(self.units.items())},
        }
        atomic_write_text(self.root / MANIFEST, json.dumps(data, indent=1, sort_keys=False) + "\n")

    def add_units(self, units: list[Unit]) -> None:
        for u in units:
            self.units.setdefault(u.key, u)
        self.save()

    def finish(
        self,
        unit: Unit,
        status: UnitStatus,
        message: str | None = None,
        peak_rss_mib: float | None = None,
    ) -> None:
        """Record a unit's outcome and save (after its files were renamed into place)."""
        unit.status, unit.message, unit.peak_rss_mib = status, message, peak_rss_mib
        self.units[unit.key] = unit
        self.save()
