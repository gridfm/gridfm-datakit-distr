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
``scenario_meta``.
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
STATE_TABLES = (
    "states/states",
    "states/yprim_state",
    "states/ybus",
    "views/kron/ybus",
)


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
            pq.write_table(data, tmp, compression=COMPRESSION)
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
        """Merge the state parts into single tables, one row set per state, with its count of
        scenarios from ``scenario_meta``; remove the parts. A part holds all rows of the states it
        met, so a state's rows are taken from the first part that has it (they are equal in every
        part); the table of an earlier consolidation (a resumed dataset) comes before the parts.
        Rows are ordered by state."""
        meta = self.read("scenarios/scenario_meta", columns=["state_id"])
        counts: dict[str, int] = {}
        if meta is not None:
            for sid in meta.column("state_id").drop_null().to_pylist():
                counts[str(sid)] = counts.get(str(sid), 0) + 1
        for table in STATE_TABLES:
            parts = sorted((self.root / table).glob("part-*.parquet"))
            if not parts:
                continue
            seen: set[str] = set()
            pieces = []
            single = self.root / f"{table}.parquet"
            sources: list[pa.Table | Path] = []
            if single.exists():  # an earlier finalisation: its rows come first
                earlier = pq.read_table(single)
                if table == "states/states":
                    earlier = earlier.drop_columns(["n_scenarios"])
                sources.append(earlier)
            sources += parts
            for source in sources:
                part = source if isinstance(source, pa.Table) else pq.read_table(source)
                known = pa.array(sorted(seen), pa.string())
                part = part.filter(pc.invert(pc.is_in(part.column("state_id"), value_set=known)))
                seen.update(str(s) for s in part.column("state_id").to_pylist())
                pieces.append(part)
            data = pa.concat_tables(pieces)
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


STATES_PART = pa.schema([f for f in schema.STATES if f.name != "n_scenarios"])
"""``states/states`` parts: without the count, known only when the dataset is complete."""
