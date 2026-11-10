"""gridfm-datakit's layout of the positive-sequence view of one network.

``views/posseq_export`` writes the view normalised to a working directory: per scenario
``views/posseq/bus_results``, per network state its buses, branches and admittance, per variant its
generator. datakit's tables repeat the static columns in every scenario and carry the branch
flows; they are rebuilt here, the flows from the voltages and the branch admittances with
datakit's formula (``S = V conj(Y V)`` on ``BASE_MVA``).

datakit's ``save.py`` (commit 1b974c0) writes ``<raw>/<table>.parquet/scenario_partition=<k>/``
Hive partitions with ``scenario`` inserted as the first column (0 .. n - 1, contiguous: graphkit
asserts it), ``k = scenario // 200``, and ``<raw>/n_scenarios.txt``; zstd, no dictionary, floats
``BYTE_STREAM_SPLIT``. Scenarios are numbered in the order of ``scenario_id`` (the seed tree),
one partition at a time, so memory stays bounded.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.writer import DatasetWriter
from gridfm_datakit_distr.views import datakit_columns as datakit
from gridfm_datakit_distr.views.posseq import BASE_MVA


def _parts(root: Path, table: str, i: int, columns: list[str] | None = None) -> pa.Table | None:
    parts = sorted((root / table).glob(f"part-{i:05d}-*.parquet"))
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in parts]) if parts else None


def rows_by_key(table: pa.Table, key: str) -> dict[str, pa.Table]:
    """The rows of each value of ``key`` (one sort, then slices; rows keep their order)."""
    if table.num_rows == 0:
        return {}
    keys = np.array([str(k) for k in table.column(key).to_pylist()])
    order = np.argsort(keys, kind="stable")
    table, keys = table.take(order), keys[order]
    values, starts = np.unique(keys, return_index=True)
    stops = np.r_[starts[1:], keys.size]
    return {
        str(v): table.slice(int(a), int(b - a))
        for v, a, b in zip(values.tolist(), starts.tolist(), stops.tolist(), strict=True)
    }


@dataclass(frozen=True, slots=True)
class ViewNetwork:
    """The normalised view's static tables, by network state and by variant."""

    buses: dict[str, pa.Table]
    branches: dict[str, pa.Table]
    ybus: dict[str, pa.Table]
    generators: dict[str, pa.Table]

    @classmethod
    def of(
        cls, buses: pa.Table, branches: pa.Table, ybus: pa.Table, generators: pa.Table
    ) -> ViewNetwork:
        return cls(
            rows_by_key(buses, "state_id"),
            rows_by_key(branches, "state_id"),
            rows_by_key(ybus, "state_id"),
            rows_by_key(generators, "variant_id"),
        )


def compose(
    results: pa.Table,
    static: ViewNetwork,
    state_of: dict[int, str],
    variant_of: dict[int, str],
    index: dict[int, int],
) -> dict[str, pa.Table]:
    """datakit's four tables (``bus``, ``gen``, ``branch``, ``y_bus``) of the scenarios of
    ``results`` (``views/posseq/bus_results`` rows), ``scenario`` = ``index[scenario_id]``."""
    sids = results.column("scenario_id").to_numpy()
    order = np.argsort(sids, kind="stable")  # each scenario's rows stay in bus order
    results, sids = results.take(order), sids[order]
    ids = np.unique(sids)
    starts = np.searchsorted(sids, ids)
    stops = np.r_[starts[1:], sids.size]
    tables: dict[str, list[pa.Table]] = {k: [] for k in datakit.FILES}
    for sid, start, stop in zip(ids.tolist(), starts.tolist(), stops.tolist(), strict=True):
        state, variant = state_of[sid], variant_of[sid]
        for kind, table in _scenario(
            index[sid], sid, results.slice(start, stop - start),
            static.buses[state], static.branches[state], static.ybus[state],
            static.generators[variant],
        ).items():  # fmt: skip
            tables[kind].append(table)
    return {kind: pa.concat_tables(parts) for kind, parts in tables.items() if parts}


def write_datakit_layout(dataset: Path, network_idx: int, raw_dir: Path, view_root: Path) -> int:
    """Write the view of network ``network_idx`` under ``raw_dir``: its tables are in
    ``view_root``, the scenarios' states and variants in ``dataset``; returns the number of
    scenarios (0: the network has no converged view).

    Streams: the parts are read in name order, which is ``scenario_id`` order (the part keys are
    the network, variant and chunk indices, and the rows of a part are in scenario order), and
    one partition of 200 scenarios is written at a time."""
    if raw_dir.exists():  # first: an export without scenarios must not leave the previous one
        shutil.rmtree(raw_dir)
    results = view_root / "views/posseq/bus_results"
    parts = [
        p
        for p in sorted(results.glob(f"part-{network_idx:05d}-*.parquet"))
        if pq.ParquetFile(p).metadata.num_rows
    ]
    columns = ["scenario_id", "state_id", "variant_id"]
    meta = _parts(dataset, "scenarios/scenario_meta", network_idx, columns)
    if not parts or meta is None:
        return 0
    writer = DatasetWriter(view_root)
    buses, branches, ybus = (
        writer.read(name)
        for name in ("views/posseq/buses", "views/posseq/branches", "views/posseq/ybus")
    )
    generators = _parts(view_root, "views/posseq/generators", network_idx)
    if buses is None or branches is None or ybus is None or generators is None:
        raise ValueError(f"{view_root}: the view's state or variant tables are missing")
    static = ViewNetwork.of(buses, branches, ybus, generators)
    meta_sids = [int(s) for s in meta.column("scenario_id").to_numpy().tolist()]
    states = [str(s) for s in meta.column("state_id").to_pylist()]
    variants = [str(v) for v in meta.column("variant_id").to_pylist()]
    state_of = dict(zip(meta_sids, states, strict=True))
    variant_of = dict(zip(meta_sids, variants, strict=True))
    size = datakit.N_SCENARIO_PER_PARTITION
    index: dict[int, int] = {}
    pending: list[pa.Table] = []
    last = -1
    for path in parts:
        table = pq.read_table(path)
        if table.num_rows == 0:
            continue
        sids = table.column("scenario_id").to_numpy()
        if sids[0] <= last or np.any(np.diff(sids) < 0):
            raise ValueError(f"{path}: scenarios out of order")
        last = int(sids[-1])
        for sid in np.unique(sids).tolist():
            index[int(sid)] = len(index)
        pending.append(table)
        pending = _flush(raw_dir, pending, static, state_of, variant_of, index, size, final=False)
    _flush(raw_dir, pending, static, state_of, variant_of, index, size, final=True)
    if index:
        (raw_dir / "n_scenarios.txt").write_text(str(len(index)), encoding="utf-8")
    return len(index)


def _flush(
    raw_dir: Path,
    pending: list[pa.Table],
    static: ViewNetwork,
    state_of: dict[int, str],
    variant_of: dict[int, str],
    index: dict[int, int],
    size: int,
    *,
    final: bool,
) -> list[pa.Table]:
    """Write every complete partition of the pending rows (and the last, incomplete one when
    ``final``); returns the rows left. Partition ``p`` holds the scenarios ``index`` places in
    ``[p size, (p + 1) size)``; the rows of a partition are a contiguous run."""
    if not pending:
        return []
    rows = pa.concat_tables(pending)
    sids = rows.column("scenario_id").to_numpy()
    unique, inverse = np.unique(sids, return_inverse=True)
    partition_of = np.array([index[int(s)] // size for s in unique.tolist()])[inverse]
    while rows.num_rows:
        partition = int(partition_of[0])
        stop = int(np.searchsorted(partition_of, partition, side="right"))
        complete = np.unique(sids[:stop]).size == size
        if stop == rows.num_rows and not (complete or final):
            break  # the partition goes on in the next part
        tables = compose(rows.slice(0, stop), static, state_of, variant_of, index)
        _write_partition(raw_dir, partition, tables)
        rows, sids, partition_of = rows.slice(stop), sids[stop:], partition_of[stop:]
    return [rows] if rows.num_rows else []


def _write_partition(raw_dir: Path, partition: int, tables: dict[str, pa.Table]) -> None:
    for kind, data in tables.items():
        floats = [f.name for f in data.schema if pa.types.is_floating(f.type)]
        directory = raw_dir / datakit.FILES[kind] / f"{datakit.PARTITION_COLUMN}={partition}"
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            data, directory / "part-0.parquet", compression="zstd", use_dictionary=False,
            column_encoding=dict.fromkeys(floats, "BYTE_STREAM_SPLIT"),
        )  # fmt: skip


def _scenario(
    index: int,
    sid: int,
    rows: pa.Table,
    buses: pa.Table,
    branches: pa.Table,
    ybus: pa.Table,
    generator: pa.Table,
) -> dict[str, pa.Table]:
    """datakit's four tables of one scenario, ``scenario`` first."""
    n, m = rows.num_rows, branches.num_rows

    def ids(k: int) -> dict[str, pa.Array]:
        return {
            datakit.SCENARIO_COLUMN: pa.array(np.full(k, index, dtype=np.int64)),
            "load_scenario_idx": pa.array(np.full(k, sid, dtype=np.int64)),
        }

    def col(table: pa.Table, name: str) -> pa.Array:
        return table.column(name).combine_chunks()

    bus = {**ids(n), "bus": col(rows, "bus")}
    for c in ("Pd", "Qd", "Pg", "Qg", "Vm", "Va"):
        bus[c] = col(rows, c)
    for c in ("PQ", "PV", "REF", "vn_kv", "min_vm_pu", "max_vm_pu", "GS", "BS"):
        bus[c] = col(buses, c)
    bus["x_scenario_id"] = pa.array(np.full(n, sid, dtype=np.int64))
    for c in ("x_truth_Vm", "x_truth_Va", "x_der_p_mw", "x_der_q_mvar", "x_der_p_avail_mw"):
        bus[c] = col(rows, c)
    v = rows.column("Vm").to_numpy() * np.exp(1j * np.radians(rows.column("Va").to_numpy()))
    f, t = branches.column("from_bus").to_numpy(), branches.column("to_bus").to_numpy()

    def y(name: str) -> np.ndarray:
        return (
            branches.column(f"{name}_r").to_numpy() + 1j * branches.column(f"{name}_i").to_numpy()
        )

    sf = v[f] * np.conj(y("Yff") * v[f] + y("Yft") * v[t]) * BASE_MVA
    st = v[t] * np.conj(y("Ytt") * v[t] + y("Ytf") * v[f]) * BASE_MVA
    branch = {**ids(m), "idx": col(branches, "idx"), "from_bus": col(branches, "from_bus"),
              "to_bus": col(branches, "to_bus"), "pf": pa.array(sf.real), "qf": pa.array(sf.imag),
              "pt": pa.array(st.real), "qt": pa.array(st.imag)}  # fmt: skip
    for c in datakit.BRANCH_COLUMNS[8:]:
        branch[c] = col(branches, c)
    branch["x_scenario_id"] = pa.array(np.full(m, sid, dtype=np.int64))
    branch["x_vector_shift_deg"] = col(branches, "x_vector_shift_deg")
    gen = {**ids(1), "idx": col(generator, "idx"), "bus": col(generator, "bus"),
           "p_mw": pa.array([float(bus["Pg"][0].as_py())]),
           "q_mvar": pa.array([float(bus["Qg"][0].as_py())])}  # fmt: skip
    for c in datakit.GEN_COLUMNS[5:]:
        gen[c] = col(generator, c)
    gen["x_scenario_id"] = pa.array([sid], pa.int64())
    k = ybus.num_rows
    yb = {**ids(k), "index1": col(ybus, "index1"), "index2": col(ybus, "index2"),
          "G": col(ybus, "G"), "B": col(ybus, "B"),
          "x_scenario_id": pa.array(np.full(k, sid, dtype=np.int64))}  # fmt: skip
    out = {}
    for kind, data in (("bus", bus), ("branch", branch), ("gen", gen), ("y_bus", yb)):
        target = schema.DATAKIT_TABLES[kind]
        expected = pa.schema(
            [pa.field(datakit.SCENARIO_COLUMN, pa.int64(), nullable=False), *target]
        )
        out[kind] = pa.table(data, schema=expected)
    return out
