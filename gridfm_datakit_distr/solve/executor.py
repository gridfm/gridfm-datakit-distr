"""Dataset generation: work units, processes x contexts, manifest and resume.

A **unit** is a chunk of scenarios of one variant: ``scenarios [c * chunk_size, ...)`` of variant
``k`` of network ``i``. Its files are named by ``(i, k, c)`` and depend on nothing else (the seed
tree; order-independent scenarios), so the dataset is the same whatever the number
of processes and contexts and whatever the order of the units.

A **task** is up to ``contexts_per_process`` units of the same variant, run by one process with
one thread and one OpenDSS context per unit. A process keeps, per thread slot, the runner and the
views of its last variant, so consecutive tasks of a variant do not rebuild them. Only the main
process writes the manifest: after each task, the status of its units.

**Resume.** A dataset directory with a manifest of the same ``config_hash`` and versions resumes:
``done`` units are skipped, the others (re)run; a unit's files are written atomically under fixed
names, so a rerun overwrites rather than duplicates. A failed unit's files are removed. Another
config needs ``settings.overwrite`` (the directory is replaced) or another dataset name.

**Finalisation**, after the units: the state tables are merged with their scenario counts, then
the dataset card and the Croissant metadata are written. (The positive-sequence view is computed
from a dataset by ``export-posseq``: ``views/posseq_export``.)
"""

from __future__ import annotations

import os
import resource
import shutil
import threading
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from gridfm_datakit_distr import __version__
from gridfm_datakit_distr.config.load import select_networks
from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_semantic import extract_semantic
from gridfm_datakit_distr.extract.source import read_source, source_tables
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.card import dataset_card, networks_table
from gridfm_datakit_distr.io.croissant import write_croissant
from gridfm_datakit_distr.io.manifest import MANIFEST, Manifest, Unit, UnitStatus, config_hash
from gridfm_datakit_distr.io.writer import STATES_PART, DatasetWriter, part_key
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.ids import variant_id
from gridfm_datakit_distr.networks.catalog import Catalog, Network, load_catalog
from gridfm_datakit_distr.networks.passport import build_passport
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.normalize.bases import s_base_va
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.solve.tables import chunk_tables, state_tables, variant_tables
from gridfm_datakit_distr.variants.generate import VariantResult, generate_variant
from gridfm_datakit_distr.variants.targets import VariantTargets, collect_targets
from gridfm_datakit_distr.views.kron import KronView
from gridfm_datakit_distr.views.posseq import PosSeqError, find_backbone

if TYPE_CHECKING:  # CasADi and the OPF modules load only with OPF labels
    from gridfm_datakit_distr.opf.labels import OPFLabeller, OPFRow


class GenerationError(RuntimeError):
    """A dataset that cannot be generated or resumed as configured."""


@dataclass(frozen=True, slots=True)
class Job:
    """A network of the dataset: its index in the seed tree and the model to load."""

    network_idx: int
    network: Network
    representation: str | None


@dataclass(frozen=True, slots=True)
class Task:
    job: Job
    variant_idx: int
    units: tuple[Unit, ...]


@dataclass(frozen=True, slots=True)
class UnitResult:
    key: str
    status: UnitStatus
    message: str | None = None
    peak_rss_mib: float | None = field(default_factory=lambda: peak_rss_mib())


def peak_rss_mib() -> float:
    """Peak resident memory of this process so far (MiB). Linux's ``VmHWM``: ``ru_maxrss`` keeps
    the parent's peak across the fork and exec of a spawned worker (measured: it grew with the
    parent's); elsewhere ``ru_maxrss``, KiB on Linux."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------------------- plan


def jobs_of(config: GenerationConfig, catalog: Catalog) -> list[Job]:
    """The networks of the dataset, in seed-tree order, with their representation's master."""
    jobs = []
    for i, selected in enumerate(select_networks(config.networks, catalog)):
        network = catalog.network(selected.network_id)
        if selected.representation is not None:
            master = network.representations[selected.representation]
            network = network.model_copy(update={"master": master})
        jobs.append(Job(i, network, selected.representation))
    return jobs


def plan_units(config: GenerationConfig, jobs: Sequence[Job]) -> list[Unit]:
    """Every unit of the dataset: chunks of ``chunk_size`` scenarios of every variant."""
    n, size = config.scenarios.n_scenarios_per_variant, config.settings.chunk_size
    return [
        Unit(job.network.id, job.network_idx, k, c, c * size, min((c + 1) * size, n))
        for job in jobs
        for k in range(config.variants.n_variants)
        for c in range((n + size - 1) // size)
    ]


def plan_tasks(units: Sequence[Unit], jobs: Sequence[Job], width: int) -> list[Task]:
    """Units grouped by variant into tasks of at most ``width`` units, in key order."""
    by_idx = {job.network_idx: job for job in jobs}
    groups: dict[tuple[int, int], list[Unit]] = {}
    for u in sorted(units, key=lambda u: u.key):
        groups.setdefault((u.network_idx, u.variant_idx), []).append(u)
    tasks = []
    for (i, k), members in groups.items():
        for start in range(0, len(members), width):
            tasks.append(Task(by_idx[i], k, tuple(members[start : start + width])))
    return tasks


def versions(config: GenerationConfig) -> dict[str, str]:
    """Generator and engine versions (they are part of a dataset's identity); CasADi's, read from
    the package metadata and never imported here, only with OPF labels."""
    found = {"generator": __version__, "opendss": DssContext.new().engine_version()}
    if config.labels.opf.enabled:
        found["casadi"] = version("casadi")
    return found


# -------------------------------------------------------------------------------------- driver


def generate(
    config: GenerationConfig,
    out: Path,
    *,
    catalog: Catalog | None = None,
    root: Path | None = None,
) -> Manifest:
    """Generate (or resume) the dataset of ``config`` in ``out``; returns its manifest."""
    catalog = catalog or load_catalog()
    jobs = jobs_of(config, catalog)
    manifest = _open(config, out, jobs)
    writer = DatasetWriter(out)
    passports = _network_tables(config, jobs, catalog, root, writer, manifest)
    manifest.add_units(plan_units(config, jobs))
    todo = [u for u in manifest.units.values() if u.status != "done"]
    width = config.settings.contexts_per_process
    tasks = plan_tasks(todo, jobs, width)
    processes = config.settings.num_processes or os.cpu_count() or 1
    units = {u.key: u for u in todo}
    if processes == 1 or not tasks:
        for task in tasks:
            _record(manifest, units, run_task(task, config, out, root, manifest.config_hash))
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
            futures = [
                pool.submit(run_task, task, config, out, root, manifest.config_hash)
                for task in tasks
            ]
            try:
                for future in as_completed(futures):
                    _record(manifest, units, future.result())
            except BaseException:  # Ctrl-C or a dead worker: stop, the queued tasks never run
                pool.shutdown(cancel_futures=True)
                raise
    finalise(config, out, manifest, passports, jobs)
    return manifest


def _open(config: GenerationConfig, out: Path, jobs: Sequence[Job]) -> Manifest:
    """The manifest of a new dataset or of the one to resume in ``out``."""
    resolved = config.model_dump(mode="json")
    current = versions(config)
    if (out / MANIFEST).exists():
        manifest = Manifest.load(out)
        same = manifest.config_hash == config_hash(resolved) and manifest.versions == current
        if same and manifest.schema_version == schema.SCHEMA_VERSION:
            return manifest
        if not config.settings.overwrite:
            raise GenerationError(
                f"{out} holds a dataset of another config or version (config_hash "
                f"{manifest.config_hash[:12]}, versions {manifest.versions}); set "
                "settings.overwrite or choose another dataset_name"
            )
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise GenerationError(f"{out} exists and is not a dataset (no {MANIFEST}): not touched")
    manifest = Manifest.create(out, resolved, current)
    manifest.networks = [
        {
            "network_id": job.network.id,
            "network_idx": job.network_idx,
            "representation": job.representation,
        }
        for job in jobs
    ]
    manifest.save()
    return manifest


def _network_tables(
    config: GenerationConfig,
    jobs: Sequence[Job],
    catalog: Catalog,
    root: Path | None,
    writer: DatasetWriter,
    manifest: Manifest,
) -> list[dict[str, Any]]:
    """Passports and ``catalog/networks`` (the networks are loaded once here: a network that
    does not load stops the run before any unit)."""
    options = prepare_options(config)
    passports = []
    for j, job in enumerate(jobs):
        prepared = prepare_network(job.network, options, root)
        passport = build_passport(prepared, catalog)
        writer.write_json(f"catalog/passports/{job.network.id}.json", passport)
        passports.append(passport)
        if config.labels.opf.enabled:  # the reason a network has no OPF labels, once
            from gridfm_datakit_distr.opf.support import unsupported_reason

            manifest.networks[j]["opf_not_supported"] = unsupported_reason(prepared.model)
    manifest.save()
    writer.write_table(
        "catalog/networks", networks_table(passports, manifest.networks), schema.CATALOG_NETWORKS
    )
    return passports


def _record(manifest: Manifest, units: dict[str, Unit], results: Sequence[UnitResult]) -> None:
    for r in results:
        manifest.finish(units[r.key], r.status, r.message, r.peak_rss_mib)


def finalise(
    config: GenerationConfig,
    out: Path,
    manifest: Manifest,
    passports: Sequence[dict[str, Any]],
    jobs: Sequence[Job],
) -> None:
    """State tables merged, dataset card, Croissant metadata."""
    writer = DatasetWriter(out)
    writer.consolidate_states()
    meta = writer.read("scenarios/scenario_meta", columns=["pf_status"])
    statuses: Counter[str] = Counter()
    if meta is not None:
        statuses.update(str(s) for s in meta.column("pf_status").to_pylist())
    opf = writer.read("opf/opf_meta", columns=["opf_status"])
    opf_statuses = (
        None if opf is None else Counter(str(s) for s in opf.column("opf_status").to_pylist())
    )
    writer.write_text("README.md", dataset_card(manifest, passports, statuses, opf_statuses))
    write_croissant(out)


# -------------------------------------------------------------------------------------- worker


@dataclass(slots=True)
class _Slot:
    """What one thread of a process keeps between tasks: its last variant's runner and views."""

    key: tuple[str, ...] | None = None
    variant: VariantResult | None = None
    semantic: str | None = None
    """The variant's semantic layer (JSON), extracted before its first scenario."""
    runner: VariantRunner | None = None
    kron: KronView | None = None
    opf: OPFLabeller | None = None
    backbone_owner: dict[str, str] = field(default_factory=dict)
    """The backbone bus of each bus (``variants/nodes.backbone_bus``)."""


_SLOTS: dict[int, _Slot] = {}
_TARGETS: dict[tuple[str, ...], VariantTargets] = {}
_LOCK = threading.Lock()


def run_task(
    task: Task, config: GenerationConfig, out: Path, root: Path | None, digest: str
) -> list[UnitResult]:
    """Run the units of a task, one thread and context each (process entry point)."""
    if len(task.units) == 1:
        return [_run_unit(0, task, task.units[0], config, out, root, digest)]
    with ThreadPoolExecutor(max_workers=len(task.units)) as pool:
        futures = [
            pool.submit(_run_unit, slot, task, unit, config, out, root, digest)
            for slot, unit in enumerate(task.units)
        ]
        return [f.result() for f in futures]


def _run_unit(
    slot_idx: int,
    task: Task,
    unit: Unit,
    config: GenerationConfig,
    out: Path,
    root: Path | None,
    digest: str,
) -> UnitResult:
    writer = DatasetWriter(out)
    try:
        slot = _slot(slot_idx, task, config, root)
        if slot.runner is None:
            assert slot.variant is not None
            return UnitResult(
                unit.key, "failed", f"variant failed: {'; '.join(slot.variant.failures)}"
            )
        tables = _unit_tables(slot, task, unit, config, digest)
        for name, (key, table, expected) in sorted(tables.items()):
            writer.write_part(name, key, table, expected)
        if unit.chunk == 0 and slot.semantic is not None and slot.variant is not None:
            rel = f"variants/semantic/{slot.variant.variant_id}.json"
            writer.write_text(rel, slot.semantic + "\n")
        return UnitResult(unit.key, "done")
    except Exception as exc:  # recorded, never dropped; no partial unit stays
        _remove_unit_files(out, task, unit)
        first = str(exc).splitlines()[0] if str(exc) else ""
        return UnitResult(unit.key, "failed", f"{type(exc).__name__}: {first}")


def _remove_unit_files(out: Path, task: Task, unit: Unit) -> None:
    """Every file the unit writes, of this attempt or of an earlier one interrupted before the
    manifest recorded it."""
    i, k = unit.network_idx, unit.variant_idx
    keys = [part_key(i, k, unit.chunk)]
    if unit.chunk == 0:
        keys.append(part_key(i, k))
        semantic = out / "variants" / "semantic" / f"{variant_id(task.job.network.id, k)}.json"
        semantic.unlink(missing_ok=True)
    for key in keys:
        for path in out.rglob(f"part-{key}.parquet"):
            path.unlink()


def _slot(slot_idx: int, task: Task, config: GenerationConfig, root: Path | None) -> _Slot:
    """The thread's runner and views for the task's variant, rebuilt when the variant changes."""
    with _LOCK:
        slot = _SLOTS.setdefault(slot_idx, _Slot())
    network, k = task.job.network, task.variant_idx
    options = prepare_options(config)
    # everything a variant depends on (a process may serve several datasets, e.g. in tests)
    base = (str(root), network.model_dump_json(), repr(options))
    key = (*base, config.model_dump_json(), str(k))
    if slot.key == key:
        return slot
    slot.key, slot.runner, slot.kron, slot.semantic, slot.opf = None, None, None, None, None
    slot.backbone_owner = {}
    with _LOCK:
        targets = _TARGETS.get(base)
    if targets is None:
        targets = collect_targets(prepare_network(network, options, root))
        with _LOCK:
            _TARGETS[base] = targets
    i, seed = task.job.network_idx, config.settings.seed
    variant = generate_variant(
        network, targets, config.variants, seed, k, options=options, root=root,
        der=config.der.pv,
    )  # fmt: skip
    slot.variant = variant
    if variant.status == "ok":
        runner = VariantRunner(network, variant, config, i, root=root)
        s_base = s_base_va(runner.ctx, runner.model)[0]  # the loads are still the variant's
        semantic = extract_semantic(runner.ctx, runner.model, s_base_va=s_base)
        slot.semantic = semantic.model_dump_json()
        slot.runner = runner
        if config.views.kron:
            slot.kron = KronView(runner, network, variant, root=root)
        if config.labels.opf.enabled:
            from gridfm_datakit_distr.opf.labels import OPFLabeller

            slot.opf = OPFLabeller(runner, config)
        slot.backbone_owner = _backbone_owner(runner.model)
    slot.key = key
    return slot


def _unit_tables(
    slot: _Slot, task: Task, unit: Unit, config: GenerationConfig, digest: str
) -> dict[str, tuple[str, pa.Table, pa.Schema]]:
    """Every table of a unit: name -> (part key, table, schema)."""
    runner = slot.runner
    assert runner is not None
    assert slot.variant is not None
    precision = config.settings.float_precision
    runner.begin_unit()
    if slot.kron is not None:
        slot.kron.begin_unit()
    outcomes, kron_pairs = [], []
    opf_rows: list[OPFRow] = []
    for s in range(unit.first, unit.stop):
        o = runner.run(s)
        outcomes.append(o)
        if slot.kron is not None and o.pf_status == "converged":
            state = runner.registry.states.get(o.state_id or "")
            kron_pairs.append((o, slot.kron.solve(o, state)))
        if slot.opf is not None:  # the runner's context still holds the scenario
            opf_rows.append(slot.opf.label(o))
    i, k = task.job.network_idx, task.variant_idx
    chunk_key, variant_key = part_key(i, k, unit.chunk), part_key(i, k)
    out: dict[str, tuple[str, pa.Table, pa.Schema]] = {}
    engines = versions_of(runner)
    if slot.opf is not None:
        engines["casadi"] = version("casadi")
    found = chunk_tables(
        runner, outcomes, precision=precision, config_hash=digest,
        versions=engines, element_currents=config.labels.element_currents,
    )  # fmt: skip
    if slot.opf is not None:
        found.update(slot.opf.tables(opf_rows, precision))
    for name, table in found.items():
        out[name] = (chunk_key, table, schema.SCENARIO_TABLES[name](precision))
    for name, table in state_tables(runner.new_states()).items():
        expected = STATES_PART if name == "states/states" else schema.TABLES[name]
        out[name] = (chunk_key, table, expected)
    if slot.kron is not None:
        out["views/kron/node_results"] = (
            chunk_key, slot.kron.results_table(kron_pairs), schema.KRON_NODE_RESULTS,
        )  # fmt: skip
        ybus = ybus_table(slot.kron.new_states())
        if ybus is not None:
            out["views/kron/ybus"] = (chunk_key, ybus, schema.YBUS_STATE)
    if unit.chunk == 0:  # the variant's own tables, once
        found_v = variant_tables(slot.variant)
        found_v.update(source_tables(runner.model, read_source(runner.ctx, runner.model)))
        for name, table in found_v.items():
            if name == "variants/nodes":
                table = _with_backbone(table, slot.backbone_owner)
            out[name] = (variant_key, table, schema.TABLES[name])
        if slot.kron is not None:
            out["views/kron/nodes"] = (variant_key, slot.kron.nodes_table(), schema.KRON_NODES)
    return out


def _backbone_owner(model: ElectricalModel) -> dict[str, str]:
    """The backbone bus each bus aggregates to; empty when the network has no three-phase
    backbone from its source (``backbone_bus`` is then null)."""
    try:
        return find_backbone(model).owner
    except PosSeqError:
        return {}


def _with_backbone(nodes: pa.Table, owner: dict[str, str]) -> pa.Table:
    """``variants/nodes`` with ``backbone_bus``: the bus each node aggregates to in the
    positive-sequence view (null outside the source's network)."""
    buses = nodes.column("bus").to_pylist()
    column = pa.array([owner.get(str(b)) for b in buses], pa.string())
    i = nodes.schema.get_field_index("backbone_bus")
    return nodes.set_column(i, schema.NODES.field("backbone_bus"), column)


def versions_of(runner: VariantRunner) -> dict[str, str]:
    """The versions recorded in ``scenario_meta`` (from the runner's own engine)."""
    return {"generator": __version__, "opendss": runner.ctx.engine_version()}


def ybus_table(states: dict[str, ElectricalModel]) -> pa.Table | None:
    """``YBUS_STATE`` rows of the states' network admittances (None if no state)."""
    parts = []
    for sid, model in states.items():
        y = model.ybus().tocoo()
        parts.append(
            pa.table(
                {
                    "state_id": [sid] * y.nnz,
                    "row": y.row.astype(np.int32),
                    "col": y.col.astype(np.int32),
                    "g": y.data.real,
                    "b": y.data.imag,
                },
                schema=schema.YBUS_STATE,
            )
        )
    return pa.concat_tables(parts) if parts else None
