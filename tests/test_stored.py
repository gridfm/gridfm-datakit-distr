"""Tests for reading a variant's stored truth back (``views/stored.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io.writer import part_key
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.views.stored import VariantStore
from tests.test_executor import CONFIG, catalog


def test_the_store_reads_back_what_the_generation_computed(tmp_path: Path) -> None:
    """With element currents stored in full precision, the store's ``Yprim · V`` at each
    scenario's state equals them bitwise; customers are the variant's enabled loads by name."""
    data = {**CONFIG, "labels": {"element_currents": True}}
    data["settings"] = {**CONFIG["settings"], "float_precision": "full"}
    cfg = GenerationConfig.model_validate(data)
    out = tmp_path / "out"
    generate(cfg, out, catalog=catalog(tmp_path), root=tmp_path)
    states = []
    for i in (0, 1):  # t/feeder, t/reg (regulated: one state per tap position met)
        for k in (0, 1):
            store = VariantStore(out, i, k, cfg)
            variant = store.variant()
            loads = [
                e.name for e in variant.model.elements if e.element_class == "load" and e.enabled
            ]
            assert variant.customer_elements == tuple(sorted(loads))
            seen = 0
            for chunk, scenarios in store.chunks():
                name = f"part-{part_key(i, k, chunk)}.parquet"
                currents = pq.read_table(out / "scenarios" / "element_results" / name)
                for o in scenarios:
                    rows = currents.filter(
                        pc.equal(currents["scenario_id"], pa.scalar(o.scenario_id))
                    )
                    expected = rows["i_re"].to_numpy() + 1j * rows["i_im"].to_numpy()
                    assert np.array_equal(o.i_pd, expected)
                    seen += 1
            assert seen == 5
            states.append(len(store._models))
    assert max(states) > 1  # the regulated network: one model per tap position met


def test_a_variant_without_converged_scenarios(tmp_path: Path) -> None:
    out = tmp_path / "out"
    cfg = GenerationConfig.model_validate(CONFIG)
    generate(cfg, out, catalog=catalog(tmp_path), root=tmp_path)
    meta_dir = out / "scenarios" / "scenario_meta"
    for path in meta_dir.glob(f"part-{part_key(0, 0)}-*.parquet"):
        meta = pq.read_table(path)
        failed = pa.array(["not_converged"] * meta.num_rows)
        pq.write_table(
            meta.set_column(meta.schema.get_field_index("pf_status"), "pf_status", failed), path
        )
    store = VariantStore(out, 0, 0, cfg)
    assert store.first_state() is None
    with pytest.raises(ValueError, match="no converged scenario"):
        store.variant()


def test_a_variant_without_pv(tmp_path: Path) -> None:
    """``scenarios/der`` parts without rows (no PV unit): no PV order, no PV output."""
    cfg = GenerationConfig.model_validate({**CONFIG, "der": {"pv": {"penetration": 0.0}}})
    out = tmp_path / "out"
    generate(cfg, out, catalog=catalog(tmp_path), root=tmp_path)
    store = VariantStore(out, 0, 0, cfg)
    assert store.variant().pv_generators == ()
    scenarios = [o for _, chunk in store.chunks() for o in chunk]
    assert len(scenarios) == 5
    assert all(o.pv_p_w is None and o.load_p_w.size > 0 for o in scenarios)
