"""Dataset writer.

Every file is written to a temporary name in its directory and renamed into place, so a reader or a
resumed run never sees a partial file. Tables are checked against their schema (``io/schema.py``)
before they are written. Layout (outputs.md §2): one part per unit,

    variants/<table>/part-{i:05d}-{k:04d}.parquet                 per variant
    states/<table>/part-{i:05d}-{k:04d}-{chunk:05d}.parquet       states met by a chunk
    scenarios/<table>/part-{i:05d}-{k:04d}-{chunk:05d}.parquet    per chunk

with ``i`` the network index, ``k`` the variant index. :meth:`DatasetWriter.consolidate_states`
merges the state parts (a state met by several chunks is stored once) into
``states/<table>.parquet`` with the number of scenarios of each state, counted in
``scenario_meta``; :func:`read_states` reads a state table by the same rule before that.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.manifest import atomic_write_text

COMPRESSION: Literal["zstd"] = "zstd"
COMPRESSION_LEVEL = 1
SCENARIOS_PER_ROW_GROUP = 16
"""Row groups of a scenario-level table: the rows of this many scenarios (a reader of one scenario
decodes one group; smaller groups cost more than 1 % in bytes on CSIRO n01_f1 and EDP rural)."""
STATE_TABLES = (
    "states/states",
    "states/yprim_state",
    "states/ybus",
    "views/kron/ybus",
)


def parquet_options(table_schema: pa.Schema) -> dict[str, Any]:
    """How every table is encoded: dictionaries only for string and bool columns (most float
    values are distinct: a dictionary only adds its pages), DELTA_BINARY_PACKED for the integer
    keys (``scenario_id``, ``*_idx``: sorted runs), the rest plain; zstd."""
    keys = [
        f.name
        for f in table_schema
        if pa.types.is_integer(f.type) and (f.name == "scenario_id" or f.name.endswith("_idx"))
    ]
    return {
        "compression": COMPRESSION,
        "compression_level": COMPRESSION_LEVEL,
        "use_dictionary": [
            f.name
            for f in table_schema
            if pa.types.is_string(f.type) or pa.types.is_boolean(f.type)
        ],
        "column_encoding": dict.fromkeys(keys, "DELTA_BINARY_PACKED") or None,
    }


def row_group_size(data: pa.Table) -> int | None:
    """Rows per row group: :data:`SCENARIOS_PER_ROW_GROUP` scenarios' for a table with
    ``scenario_id`` (every scenario has as many rows); pyarrow's default (one group up to 2^20
    rows) for the others."""
    if "scenario_id" not in data.column_names or data.num_rows == 0:
        return None
    scenarios = len(pc.unique(data.column("scenario_id")))
    return SCENARIOS_PER_ROW_GROUP * (data.num_rows // scenarios)


def part_key(i: int, k: int, chunk: int | None = None) -> str:
    """``{i:05d}-{k:04d}`` or ``{i:05d}-{k:04d}-{chunk:05d}``."""
    return f"{i:05d}-{k:04d}" if chunk is None else f"{i:05d}-{k:04d}-{chunk:05d}"


class DatasetWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def write_part(self, table: str, key: str, data: pa.Table, expected: pa.Schema) -> str:
        """Write ``data`` as ``<table>/part-<key>.parquet``; returns its path from the root."""
        schema.check_schema(data, expected, table)
        rel = f"{table}/part-{key}.parquet"
        self._write(rel, data)
        return rel

    def write_table(self, table: str, data: pa.Table, expected: pa.Schema) -> str:
        """Write a single-file table ``<table>.parquet``."""
        schema.check_schema(data, expected, table)
        rel = f"{table}.parquet"
        self._write(rel, data)
        return rel

    def write_json(self, rel: str, obj: Any) -> str:
        atomic_write_text(self.root / rel, json.dumps(obj, indent=1, sort_keys=True) + "\n")
        return rel

    def write_text(self, rel: str, text: str) -> str:
        atomic_write_text(self.root / rel, text)
        return rel

    def _write(self, rel: str, data: pa.Table) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            pq.write_table(
                data, tmp, row_group_size=row_group_size(data), **parquet_options(data.schema)
            )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def read(self, table: str, columns: list[str] | None = None) -> pa.Table | None:
        """All parts of a table (or its single file), None if absent; only ``columns`` if
        given (a dataset's scenario tables can be far larger than memory)."""
        single = self.root / f"{table}.parquet"
        if single.exists():
            return pq.read_table(single, columns=columns)
        parts = sorted((self.root / table).glob("part-*.parquet"))
        if not parts:
            return None
        return pa.concat_tables([pq.read_table(p, columns=columns) for p in parts])

    def consolidate_states(self) -> None:
        """Merge the state parts into single tables (:func:`read_states`: one row set per state,
        that of the first file holding it), with each state's count of scenarios from
        ``scenario_meta``; remove the parts. Rows are ordered by state."""
        meta = self.read("scenarios/scenario_meta", columns=["state_id"])
        counts: dict[str, int] = {}
        if meta is not None:
            for sid in meta.column("state_id").drop_null().to_pylist():
                counts[str(sid)] = counts.get(str(sid), 0) + 1
        for table in STATE_TABLES:
            if not any((self.root / table).glob("part-*.parquet")):
                continue
            # an earlier finalisation's count is recomputed below
            columns = STATES_PART.names if table == "states/states" else None
            data = read_states(self.root, table, columns=columns)
            assert data is not None
            # by state, whatever the history of the parts (a resumed run); the sort is stable
            data = data.take(pc.sort_indices(data, sort_keys=[("state_id", "ascending")]))
            if table == "states/states":
                ids = [str(s) for s in data.column("state_id").to_pylist()]
                n = pa.array([counts.get(s, 0) for s in ids], pa.int32())
                data = data.append_column(schema.STATES.field("n_scenarios"), n)
                expected = schema.STATES
            else:
                expected = schema.TABLES[table]
            self.write_table(table, data, expected)
            shutil.rmtree(self.root / table)  # with any temporary file of a killed writer


def read_states(
    root: Path, table: str, filters: Any = None, columns: list[str] | None = None
) -> pa.Table | None:
    """A state table as :meth:`DatasetWriter.consolidate_states` merges it, from whatever is on
    disk: the single file of an earlier consolidation, then the parts in name order, each state's
    rows taken from the first of them that holds it (a part holds all rows of the states its chunk
    met, equal in every part). Rows keep their file's order; ``filters`` (pyarrow's) and
    ``columns`` apply to every file. None if the table has no file."""
    single = root / f"{table}.parquet"
    sources = [single] if single.exists() else []
    sources += sorted((root / table).glob("part-*.parquet"))
    seen: set[str] = set()
    pieces = []
    for path in sources:
        part = pq.read_table(path, filters=filters, columns=columns)
        ids = part.column("state_id")
        # membership against the part's own ids: against all ids seen it grows with the parts
        new = [str(s) for s in pc.unique(ids).to_pylist() if s not in seen]
        pieces.append(part.filter(pc.is_in(ids, value_set=pa.array(new, pa.string()))))
        seen.update(new)
    return pa.concat_tables(pieces) if pieces else None


STATES_PART = pa.schema([f for f in schema.STATES if f.name != "n_scenarios"])
"""``states/states`` parts: without the count, known only when the dataset is complete."""
