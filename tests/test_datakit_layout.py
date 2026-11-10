"""Tests for the positive-sequence export in datakit's layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.io.writer import DatasetWriter
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.views import datakit_columns as datakit
from gridfm_datakit_distr.views import posseq_export
from gridfm_datakit_distr.views.datakit_layout import ViewNetwork, compose, rows_by_key
from tests.test_executor import catalog, config


def test_rows_by_key_keeps_row_order_and_takes_empty_tables() -> None:
    table = pa.table({"k": ["b", "a", "b", "a"], "x": [0, 1, 2, 3]})
    split = rows_by_key(table, "k")
    assert {k: v.column("x").to_pylist() for k, v in split.items()} == {"a": [1, 3], "b": [0, 2]}
    assert rows_by_key(table.slice(0, 0), "k") == {}


def test_partitions_stream_across_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Partitions of 3 scenarios over parts of 2 (chunk size): the streamed export equals the
    composition of every scenario at once."""
    out = tmp_path / "out"
    generate(config(), out, catalog=catalog(tmp_path), root=tmp_path)  # 2 variants x 5 scenarios
    monkeypatch.setattr(datakit, "N_SCENARIO_PER_PARTITION", 3)
    [result] = posseq_export.export_posseq(out, tmp_path / "export", ["t/feeder"])
    raw = result.raw_dir
    assert result.scenarios == 10
    names = sorted(p.name for p in (raw / "bus_data.parquet").iterdir())
    assert names == [f"scenario_partition={k}" for k in range(4)]  # 3 + 3 + 3 + 1
    streamed = {}
    for kind, directory in datakit.FILES.items():
        tables = [
            pq.read_table(raw / directory / f"scenario_partition={k}" / "part-0.parquet")
            for k in range(4)
        ]
        for k, table in enumerate(tables):
            assert set((table.column("scenario").to_numpy() // 3).tolist()) == {k}
        streamed[kind] = pa.concat_tables(tables)
    # everything at once: the view of each variant, composed together
    work = tmp_path / "work"
    for k in posseq_export._variant_indices(out, 0):
        posseq_export._variant_task((out, work, 0, k))
    view = DatasetWriter(work)
    results = view.read("views/posseq/bus_results")
    assert results is not None
    meta = pa.concat_tables(
        [pq.read_table(p) for p in sorted((out / "scenarios/scenario_meta").glob("part-00000-*"))]
    )
    sids = meta.column("scenario_id").to_pylist()
    state_of = dict(zip(sids, meta.column("state_id").to_pylist(), strict=True))
    variant_of = dict(zip(sids, meta.column("variant_id").to_pylist(), strict=True))
    tables = [view.read(f"views/posseq/{n}") for n in ("buses", "branches", "ybus", "generators")]
    static = ViewNetwork.of(*tables)  # type: ignore[arg-type]
    ids = np.unique(results.column("scenario_id").to_numpy()).tolist()
    index = {sid: k for k, sid in enumerate(ids)}
    whole = compose(results, static, state_of, variant_of, index)
    for kind in datakit.FILES:
        assert streamed[kind].equals(whole[kind]), kind
