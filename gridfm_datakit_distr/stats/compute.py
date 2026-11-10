"""Statistics of a generated dataset.

The summary reads only the manifest and the scenario-level tables: units by status, scenarios by
power-flow status per network, the views' statuses, solve times and the size on disk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gridfm_datakit_distr.io.manifest import Manifest
from gridfm_datakit_distr.io.writer import DatasetWriter


@dataclass(slots=True)
class Summary:
    name: str
    schema_version: str
    units: Counter[str]
    scenarios: dict[str, Counter[str]]
    """network_id -> pf_status -> count."""
    views: dict[str, Counter[str]] = field(default_factory=dict)
    """``kron`` -> view_status -> count (one per scenario)."""
    solve_ms: dict[str, tuple[float, float]] = field(default_factory=dict)
    """network_id -> median and 95th percentile of ``t_solve_s`` (ms), converged scenarios."""
    bytes_by_dir: dict[str, int] = field(default_factory=dict)


def summary(dataset: Path) -> Summary:
    """The summary of the dataset in ``dataset``."""
    manifest = Manifest.load(dataset)
    writer = DatasetWriter(dataset)
    units: Counter[str] = Counter(str(u.status) for u in manifest.units.values())
    scenarios: dict[str, Counter[str]] = {}
    solve_ms: dict[str, tuple[float, float]] = {}
    meta = writer.read("scenarios/scenario_meta", columns=["network_id", "pf_status", "t_solve_s"])
    if meta is not None:
        networks = np.array([str(n) for n in meta.column("network_id").to_pylist()])
        status = np.array([str(s) for s in meta.column("pf_status").to_pylist()])
        seconds = np.array(meta.column("t_solve_s").to_pylist(), dtype=np.float64)
        for network_id in sorted(set(networks.tolist())):
            mine = networks == network_id
            scenarios[network_id] = Counter(status[mine].tolist())
            t = seconds[mine & (status == "converged")] * 1e3
            if t.size:
                solve_ms[network_id] = (float(np.median(t)), float(np.percentile(t, 95)))
    views: dict[str, Counter[str]] = {}
    statuses_kron: Counter[str] = Counter()
    for part in sorted((dataset / "views" / "kron" / "node_results").glob("part-*.parquet")):
        table = pq.read_table(part, columns=["scenario_id", "view_status"])  # rows per node
        ids = table.column("scenario_id").to_numpy()
        first = np.unique(ids, return_index=True)[1]
        statuses_kron.update(str(s) for s in table.column("view_status").take(first).to_pylist())
    if statuses_kron:
        views["kron"] = statuses_kron
    sizes: Counter[str] = Counter()
    for path in dataset.rglob("*"):
        if path.is_file():
            sizes[path.relative_to(dataset).parts[0]] += path.stat().st_size
    return Summary(
        name=str(manifest.config.get("dataset_name", dataset.name)),
        schema_version=manifest.schema_version,
        units=units,
        scenarios=scenarios,
        views=views,
        solve_ms=solve_ms,
        bytes_by_dir=dict(sorted(sizes.items())),
    )


def summary_text(s: Summary) -> str:
    """The summary as plain text."""

    def counts(c: Counter[str]) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "none"

    total = sum((c for c in s.scenarios.values()), Counter())
    lines = [
        f"dataset {s.name} (schema {s.schema_version})",
        f"units: {counts(s.units)}",
        f"scenarios: {sum(total.values())} ({counts(total)})",
    ]
    for network_id, c in s.scenarios.items():
        timing = ""
        if network_id in s.solve_ms:
            median, p95 = s.solve_ms[network_id]
            timing = f"; solve {median:.1f} ms median, {p95:.1f} ms p95"
        lines.append(f"  {network_id}: {counts(c)}{timing}")
    for view, c in s.views.items():
        lines.append(f"view {view}: {counts(c)}")
    size = sum(s.bytes_by_dir.values())
    lines.append(
        f"size: {size / 1e6:.1f} MB ("
        + ", ".join(f"{d} {b / 1e6:.1f}" for d, b in s.bytes_by_dir.items())
        + ")"
    )
    return "\n".join(lines) + "\n"
