"""Throughput and storage budgets as a benchmark; run on demand with ``-m perf``.

The budgets were measured on 10⁵ snapshots of 50 CSIRO feeders. This benchmark checks the
same two hard limits on the median CSIRO feeder, so that a change breaking
them fails a test instead of waiting for the next measurement. Storage is deterministic for given
engine and library versions; time depends on the machine, and the 24 h limit leaves a wide margin
(0.26 h without views).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.solve.executor import generate

pytestmark = [pytest.mark.perf, pytest.mark.network_data, pytest.mark.slow]

NETWORK = "csiro_enwl/n21_f3"
"""The median CSIRO feeder (3376 nodes)."""
MEAN_NODES = 4440
"""Nodes of the mean CSIRO feeder: the budget's reference."""
SNAPSHOTS = 1e6
STORAGE_LIMIT = 150e9
"""Bytes per 10⁶ snapshots (hard limit)."""
HOURS_LIMIT, CORES = 24.0, 32
SCENARIOS = 500


def test_budgets_on_the_median_feeder(
    tmp_path: Path, network_master: Callable[[str], Path]
) -> None:
    """Default config, one variant, one chunk of 500 scenarios, one process; bytes are
    every file of the dataset, time is the wall clock of ``generate`` (set-up included)."""
    network_master(NETWORK)  # skips without the CSIRO data
    config = GenerationConfig.model_validate(
        {
            "dataset_name": "budget",
            "networks": {"include": [NETWORK]},
            "der": {"pv": {"penetration": {"distribution": "uniform", "low": 0.0, "high": 0.6}}},
            "scenarios": {"n_scenarios_per_variant": SCENARIOS},
            "settings": {"seed": 19, "chunk_size": SCENARIOS, "num_processes": 1},
        }
    )
    out = tmp_path / "d"
    start = time.perf_counter()
    generate(config, out)
    seconds = time.perf_counter() - start
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    nodes = sum(pq.read_table(out / "catalog" / "networks.parquet").column("n_nodes").to_pylist())
    stored = size / (nodes * SCENARIOS) * MEAN_NODES * SNAPSHOTS
    hours = SNAPSHOTS * seconds / SCENARIOS / CORES / 3600
    print(f"\n{stored / 1e9:.0f} GB and {hours:.2f} h per 10^6 snapshots on {CORES} cores")
    assert stored <= STORAGE_LIMIT
    assert hours <= HOURS_LIMIT
