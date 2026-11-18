"""Reference loader and closed loop: the stored tables determine the stored voltages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.reader import DatasetReader
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.validate.closed_loop import closed_loop, linearised, no_load
from tests.test_executor import catalog, config


@pytest.fixture(scope="module", params=["full", "mixed"])
def dataset(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp(request.param)
    generate(config(float_precision=request.param), root / "out", catalog=catalog(root), root=root)
    return root / "out"


def test_the_source_tables(dataset: Path) -> None:
    source = pq.read_table(dataset / "variants" / "source")
    schema.check_schema(source, schema.SOURCE, "variants/source")
    y = pq.read_table(dataset / "variants" / "source_yprim")
    schema.check_schema(y, schema.SOURCE_YPRIM, "variants/source_yprim")
    variants = set(pq.read_table(dataset / "variants" / "variants")["variant_id"].to_pylist())
    assert set(source["variant_id"].to_pylist()) == variants
    for vid in variants:  # three conductors at 1 pu of the source's base voltage
        rows = [r for r in source.to_pylist() if r["variant_id"] == vid]
        assert [r["position"] for r in rows] == [0, 1, 2]
        e = np.array([complex(r["e_re"], r["e_im"]) for r in rows])
        assert np.allclose(e[1:] / e[:-1], np.exp(-2j * np.pi / 3))


def test_the_closed_loop_reproduces_the_stored_voltages(dataset: Path) -> None:
    reader = DatasetReader(dataset)
    meta = reader.meta.to_pylist()
    ids = [r["scenario_id"] for r in meta if r["pf_status"] == "converged"]
    assert len(ids) >= len(meta) // 2
    for sc in reader.scenarios(ids):
        result = closed_loop(sc)
        assert result.max_dv_pu <= 1e-6, (sc.variant.variant_id, result)
        assert sc.v is not None
        base = sc.variant.base_v
        err_lin = np.max(np.abs(linearised(sc) - sc.v) / base)
        err_none = np.max(np.abs(no_load(sc) - sc.v) / base)
        assert result.max_dv_pu < err_lin < err_none


def test_the_reader_imports_no_engine() -> None:
    code = (
        "import sys, gridfm_datakit_distr.io.reader, gridfm_datakit_distr.validate.closed_loop\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('opendssdirect', 'dss', 'casadi')"
        " or m.startswith('gridfm_datakit_distr.engine')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
