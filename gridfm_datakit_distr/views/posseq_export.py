"""``export-posseq``: the positive-sequence view of a dataset in gridfm-datakit's layout.

The view is computed after generation from the stored truth, so that a dataset holds the
three-phase model only. For each network:

1. each variant (in parallel processes) is read back (``views/stored.py``), its view built and
   solved for every converged scenario (``views/posseq.py``), and the view's tables written to
   ``<out>/<network_id>/.posseq-work/`` (per scenario, per network state, per variant);
2. ``views/datakit_layout.py`` writes ``<out>/<network_id>/raw/`` from them, one partition of
   scenarios at a time, and the working directory is removed.

Next to ``raw/`` (graphkit's ``data_path``) stay ``posseq_scenarios.parquet``, the status of the
view of every scenario whose full model converged (a view that fails is recorded, never dropped),
and ``posseq_mapping.parquet``, the buses of the full model that each bus of
the view aggregates.
"""

from __future__ import annotations

import multiprocessing
import shutil
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gridfm_datakit_distr.config.load import stored_config
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.manifest import MANIFEST, Manifest
from gridfm_datakit_distr.io.writer import COMPRESSION, DatasetWriter, part_key
from gridfm_datakit_distr.networks.catalog import safe_relative_path
from gridfm_datakit_distr.views.datakit_layout import write_datakit_layout
from gridfm_datakit_distr.views.posseq import PosSeqView
from gridfm_datakit_distr.views.stored import VariantStore

WORK_DIR = ".posseq-work"
STATUS_FILE = "posseq_scenarios.parquet"
MAPPING_FILE = "posseq_mapping.parquet"


@dataclass(frozen=True, slots=True)
class NetworkExport:
    network_id: str
    raw_dir: Path
    scenarios: int
    """Scenarios written to ``raw_dir``: those whose view converged."""
    statuses: dict[str, int]
    """View status -> scenarios, over every scenario whose full model converged."""


def export_posseq(
    dataset: Path, out: Path, networks: Sequence[str] | None = None, processes: int = 1
) -> list[NetworkExport]:
    """Export the view of ``networks`` (default: every network of the dataset) under ``out``."""
    manifest = Manifest.load(dataset)
    known = {str(n["network_id"]): int(n["network_idx"]) for n in manifest.networks}
    for network_id in known:  # each id becomes a folder under `out`; a dataset is outside input
        try:
            safe_relative_path(network_id)
        except ValueError as exc:
            raise ValueError(f"{dataset / MANIFEST}: {exc}") from exc
    chosen = list(networks) if networks else sorted(known, key=lambda n: known[n])
    unknown = [n for n in chosen if n not in known]
    if unknown:
        raise ValueError(f"not in the dataset: {', '.join(unknown)}")
    # The view is built from the files on disk: a failed or unfinished unit would drop silently.
    incomplete = sorted(
        {u.network_id for u in manifest.units.values() if u.status != "done"} & set(chosen)
    )
    if incomplete:
        raise ValueError(
            f"incomplete in the dataset (failed or unfinished units; resume generate): "
            f"{', '.join(incomplete)}"
        )
    return [_network(dataset, out, n, known[n], processes) for n in chosen]


def _network(dataset: Path, out: Path, network_id: str, i: int, processes: int) -> NetworkExport:
    target = out / network_id
    work = target / WORK_DIR
    shutil.rmtree(work, ignore_errors=True)
    tasks = [(dataset, work, i, k) for k in _variant_indices(dataset, i)]
    if processes > 1 and len(tasks) > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(min(processes, len(tasks)), mp_context=context) as pool:
            list(pool.map(_variant_task, tasks))
    else:
        for task in tasks:
            _variant_task(task)
    raw = target / "raw"
    n = write_datakit_layout(dataset, i, raw, work)
    statuses = _collect(work, "views/posseq/scenarios", target / STATUS_FILE)
    _collect(work, "views/posseq/mapping", target / MAPPING_FILE)
    shutil.rmtree(work, ignore_errors=True)
    counts = Counter(str(s) for s in statuses.column("view_status").to_pylist())
    return NetworkExport(network_id, raw, n, dict(sorted(counts.items())))


def _variant_indices(dataset: Path, i: int) -> list[int]:
    """Indices of the variants of network ``i`` that the dataset holds."""
    parts = sorted((dataset / "variants/variants").glob(f"part-{i:05d}-*.parquet"))
    return [int(p.stem.split("-")[2]) for p in parts]


def _variant_task(task: tuple[Path, Path, int, int]) -> None:
    """Build and solve the view of variant ``k`` of network ``i``; write its tables to ``work``
    (process entry point)."""
    dataset, work, i, k = task
    config = stored_config(Manifest.load(dataset).config)
    store = VariantStore(dataset, i, k, config)
    if store.first_state() is None:
        return  # no scenario converged: nothing to export
    writer = DatasetWriter(work)
    try:
        view = PosSeqView(store.variant(), config)
    except Exception as exc:  # e.g. no three-phase backbone: recorded per scenario
        message = f"{type(exc).__name__}: {exc}"
        for chunk, sids in store.converged():
            failed = pa.table(
                {
                    "scenario_id": pa.array(sids, pa.int64()),
                    "variant_id": [store.variant_id] * len(sids),
                    "view_status": ["error"] * len(sids),
                    "view_message": [message] * len(sids),
                },
                schema=schema.POSSEQ_SCENARIOS,
            )
            name = "views/posseq/scenarios"
            writer.write_part(name, part_key(i, k, chunk), failed, schema.POSSEQ_SCENARIOS)
        return
    view.begin_unit()  # the state tables then cover every state of the variant, once
    for chunk, scenarios in store.chunks():
        pairs = [(o, view.solve(o)) for o in scenarios]
        for name, table in view.tables(pairs).items():
            writer.write_part(name, part_key(i, k, chunk), table, schema.POSSEQ_TABLES[name])
    variant_key = part_key(i, k)
    for name, table in view.state_tables().items():
        writer.write_part(name, variant_key, table, schema.POSSEQ_TABLES[name])
    for name, table in (
        ("views/posseq/mapping", view.mapping_table()),
        ("views/posseq/generators", view.generators_table()),
    ):
        writer.write_part(name, variant_key, table, schema.POSSEQ_TABLES[name])


def _collect(work: Path, table: str, path: Path) -> pa.Table:
    """The parts of ``table`` as one file at ``path`` (empty if no part)."""
    data = DatasetWriter(work).read(table)
    if data is None:
        data = schema.POSSEQ_TABLES[table].empty_table()
    pq.write_table(data, path, compression=COMPRESSION)
    return data
