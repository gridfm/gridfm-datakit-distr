"""Shared pytest configuration: opt-in markers.

- ``julia`` tests run only with ``DISTR_RUN_JULIA=1`` (they need Julia and PMD).
- ``perf`` tests run only when selected explicitly with ``-m perf`` (or ``DISTR_RUN_PERF=1``).
- ``network_data`` tests get their files through the ``network_master`` fixture, which skips
  them when the source is not in the cache, or fails them with ``DISTR_REQUIRE_NETWORK_DATA=1``
  (CI, where every source is fetched).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_julia = _env_flag("DISTR_RUN_JULIA")
    run_perf = _env_flag("DISTR_RUN_PERF") or "perf" in (config.option.markexpr or "")
    skip_julia = pytest.mark.skip(reason="needs Julia/PMD: set DISTR_RUN_JULIA=1")
    skip_perf = pytest.mark.skip(reason="benchmark: select with -m perf")
    for item in items:
        if "julia" in item.keywords and not run_julia:
            item.add_marker(skip_julia)
        if "perf" in item.keywords and not run_perf:
            item.add_marker(skip_perf)


@pytest.fixture
def network_master() -> Callable[[str], Path]:
    """Master file of a catalog network in the cache; skips the test if it was not fetched."""
    from gridfm_datakit_distr.networks.cache import is_fetched, master_path
    from gridfm_datakit_distr.networks.catalog import load_catalog

    def get(network_id: str) -> Path:
        catalog = load_catalog()
        network = catalog.network(network_id)
        source = catalog.source(network.source)
        if not is_fetched(source):
            missing = f"network data missing: run `gridfm_datakit_distr fetch {source.id}`"
            if _env_flag("DISTR_REQUIRE_NETWORK_DATA"):
                pytest.fail(missing)
            pytest.skip(missing)
        return master_path(network)

    return get
