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
]


def make_provider(
    config: Profiles, customers: Sequence[Customer], network: Network, root: Path | None = None
) -> ProfileProvider:
    """The provider ``config.provider`` for a variant's customers."""
    if config.provider == "synthetic":
        return SyntheticProvider(customers, config.synthetic)
    if config.provider in ("network_loadshapes", "edp_smart_meter"):
        return LoadshapeProvider(
            customers,
            read_series(network, root),
            same_timestamp=config.same_timestamp,
            name=config.provider,
        )
    raise ProfileError(f"provider {config.provider!r} is not available in this version")
