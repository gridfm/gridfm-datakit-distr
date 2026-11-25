"""Load profile providers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from gridfm_datakit_distr.config.schema import Profiles
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.scenarios.profiles.base import (
    Customer,
    LoadDraw,
    ProfileError,
    ProfileProvider,
    customers_of,
)
from gridfm_datakit_distr.scenarios.profiles.network_loadshapes import (
    LoadshapeProvider,
    read_series,
)
from gridfm_datakit_distr.scenarios.profiles.synthetic import SyntheticProvider

__all__ = [
    "Customer",
    "LoadDraw",
    "ProfileError",
    "ProfileProvider",
    "customers_of",
    "make_provider",
    "profile_issues",
]


def make_provider(
    config: Profiles,
    customers: Sequence[Customer],
    network: Network,
    root: Path | None = None,
    level: float = 1.0,
) -> ProfileProvider:
    """The provider ``config.provider`` for a variant's customers; ``level``, the variant's load
    level, scales the synthetic provider's nominal loads (drawn for it only)."""
    if config.provider == "synthetic":
        return SyntheticProvider(customers, config.synthetic, level)
    if config.provider in ("network_loadshapes", "edp_smart_meter"):
        return LoadshapeProvider(
            customers,
            read_series(network, root),
            same_timestamp=config.same_timestamp,
            name=config.provider,
        )
    raise ProfileError(f"provider {config.provider!r} is not available in this version")


def profile_issues(
    config: Profiles, networks: Sequence[Network], root: Path | None = None
) -> list[str]:
    """Why ``config.provider`` cannot serve each network it cannot serve, before any work: a
    loadshape provider reads each network's loadshapes as :func:`make_provider` does."""
    if config.provider == "synthetic":
        return []
    issues = []
    for network in networks:
        try:
            read_series(network, root)
        except ProfileError as exc:
            issues.append(str(exc))
    return issues
