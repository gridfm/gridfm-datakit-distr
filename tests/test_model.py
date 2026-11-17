"""Tests for the electrical model, its identifiers and its Parquet round trip."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.writer import DatasetWriter
from gridfm_datakit_distr.model import io_parquet
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element, Node
from gridfm_datakit_distr.model.ids import base_variant_id, split_variant_id, variant_id

Y_LINE = np.array([[2 - 4j, -(2 - 4j)], [-(2 - 4j), 2 - 4j]])  # 1-conductor series branch
Y_CAP = np.array([[0.5j, -0.5j], [-0.5j, 0.5j]])  # capacitor: terminal 2 grounded
Y_TR = np.array([[10 - 30j, -(10 - 30j)], [-(10 - 30j), 10 - 30j]])


def small_model(variant: str = "net/v0000") -> ElectricalModel:
    """src --line-- n1 --transformer(tap-dependent)-- n2, capacitor at n1, load at n2."""
    nodes = [
        Node(0, "src", "a", "src.1", 0.23, is_source=True),
        Node(1, "n1", "a", "n1.1", 0.23),
        Node(2, "n2", "a", "n2.1", 0.23, is_grounded=True, r_ground_ohm=5.0, x_ground_ohm=0.0),
    ]
    elements = [
        Element(0, "Line.l1", "line", True, True, [[0], [1]], Y_LINE.copy(), rating_a=400.0),
        Element(1, "Capacitor.c1", "capacitor", True, True, [[1], [GROUND]], Y_CAP.copy()),
        Element(2, "Transformer.t1", "transformer", True, True, [[1], [2]], Y_TR.copy(),
                is_tap_dependent=True, rating_va=250e3),
        Element(3, "Load.ld", "load", False, True, [[2, GROUND]]),
        Element(4, "Line.spare", "line", True, False, [[1], [2]]),
    ]  # fmt: skip
    return ElectricalModel("net", variant, nodes, elements)


# ---------------------------------------------------------------- ids


def test_variant_ids() -> None:
    assert variant_id("csiro_enwl/n01_f1", 7) == "csiro_enwl/n01_f1/v0007"
    assert split_variant_id("csiro_enwl/n01_f1/v0007") == ("csiro_enwl/n01_f1", "v0007")
    assert split_variant_id(base_variant_id("ieee13")) == ("ieee13", "base")
    with pytest.raises(ValueError, match="variant index"):
        variant_id("ieee13", 10_000)
    with pytest.raises(ValueError, match="not a variant id"):
        split_variant_id("ieee13")


# ---------------------------------------------------------------- admittance and currents


def test_ybus_of_a_small_model() -> None:
    model = small_model()
    y = model.ybus().toarray()
    expected = np.zeros((3, 3), dtype=complex)
    expected[np.ix_([0, 1], [0, 1])] += Y_LINE
    expected[1, 1] += Y_CAP[0, 0]
    expected[np.ix_([1, 2], [1, 2])] += Y_TR
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(model.source_nodes(), [0])


def test_element_blocks_give_element_currents() -> None:
    model = small_model()
    yblk, selection = model.element_blocks()
    v = np.array([230 + 0j, 228 - 2j, 226 - 4j])
    currents = yblk @ (selection @ v)
    expected = np.concatenate([Y_LINE @ v[[0, 1]], Y_CAP @ np.array([v[1], 0]), Y_TR @ v[[1, 2]]])
    np.testing.assert_allclose(currents, expected, rtol=1e-15)
    assert selection.shape == (6, 3)
    assert selection[3].nnz == 0  # the grounded capacitor terminal


def test_model_validation() -> None:
    node = Node(0, "a", "a", "a.1", 0.23)
    with pytest.raises(ValueError, match="expected 0"):
        ElectricalModel("n", "n/base", [Node(1, "a", "a", "a.1", 0.23)], [])
    bad = Element(0, "Line.x", "line", True, True, [[0], [3]], Y_LINE)
    with pytest.raises(ValueError, match="out of range"):
        ElectricalModel("n", "n/base", [node], [bad])
    with pytest.raises(ValueError, match="complex128"):
        Element(0, "Line.x", "line", True, True, [[0], [0]], np.eye(3, dtype=complex))
    with pytest.raises(ValueError, match="different numbers"):
        Element(0, "Line.x", "line", True, True, [[0], [0, 0]])
    no_yprim = Element(0, "Line.x", "line", True, True, [[0], [0]])
    model = ElectricalModel("n", "n/base", [node], [no_yprim])
    with pytest.raises(ValueError, match="without Yprim"):
        model.ybus()


@st.composite
def random_models(draw: st.DrawFn) -> ElectricalModel:
    n = draw(st.integers(1, 6))
    values = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
    elements = []
    for k in range(draw(st.integers(0, 6))):
        n_cond = draw(st.integers(1, 3))
        terminals = [[draw(st.integers(GROUND, n - 1)) for _ in range(n_cond)] for _ in range(2)]
        shape = (2 * n_cond, 2 * n_cond)
        re = draw(arrays(np.float64, shape, elements=values))
        im = draw(arrays(np.float64, shape, elements=values))
        y = re + 1j * im
        elements.append(Element(k, f"Line.l{k}", "line", True, True, terminals, y))
    nodes = [Node(i, f"b{i}", "a", f"b{i}.1", 0.23) for i in range(n)]
    return ElectricalModel("net", "net/v0000", nodes, elements)


@settings(max_examples=200, deadline=None)
@given(random_models())
def test_ybus_equals_dense_assembly(model: ElectricalModel) -> None:
    dense = np.zeros((model.n_nodes, model.n_nodes), dtype=complex)
    for element in model.pd_elements():
        assert element.yprim is not None
        local = element.local_nodes()
        for a, node_a in enumerate(local):
            for b, node_b in enumerate(local):
                if node_a != GROUND and node_b != GROUND:
                    dense[node_a, node_b] += element.yprim[a, b]
    np.testing.assert_allclose(model.ybus().toarray(), dense, rtol=1e-12, atol=1e-9)


# ---------------------------------------------------------------- Parquet round trip


def test_parquet_round_trip(tmp_path: Path) -> None:
    model = small_model()
    tables = io_parquet.to_tables(model, state_id="s1")
    names = (io_parquet.NODES, io_parquet.ELEMENTS, io_parquet.TERMINALS, io_parquet.YPRIM)
    assert set(tables) == {*names, io_parquet.YPRIM_STATE}  # a subset of the dataset's tables
    writer = DatasetWriter(tmp_path)
    for name, table in tables.items():
        writer.write_table(name, table, schema.TABLES[name])  # schemas checked
    read = {name: pq.read_table(tmp_path / f"{name}.parquet") for name in tables}
    back = io_parquet.from_tables(read, model.variant_id, state_id="s1")
    assert back == model
    np.testing.assert_array_equal(back.ybus().toarray(), model.ybus().toarray())


def test_tap_dependent_yprim_needs_its_state(tmp_path: Path) -> None:
    model = small_model()
    with pytest.raises(ValueError, match="state_id is required"):
        io_parquet.to_tables(model)
    tables = io_parquet.to_tables(model, state_id="s1")
    variant_only = {k: v for k, v in tables.items() if k != io_parquet.YPRIM_STATE}
    back = io_parquet.from_tables(variant_only, model.variant_id)
    assert back.elements[2].yprim is None
    with pytest.raises(ValueError, match="without Yprim"):
        back.ybus()


def test_a_state_yprim_of_zeros_reads_back_as_zeros() -> None:
    """An element whose Yprim at the state is all zero (a switched capacitor with every step
    open) has no stored entries: with its state given it reads back as zeros, not as missing."""
    model = small_model()
    model.elements[2].yprim = np.zeros_like(model.elements[2].yprim)
    tables = io_parquet.to_tables(model, state_id="s1")
    assert tables[io_parquet.YPRIM_STATE].num_rows == 0
    back = io_parquet.from_tables(tables, model.variant_id, "s1")
    np.testing.assert_array_equal(back.elements[2].yprim, model.elements[2].yprim)
    back.ybus()  # no element without Yprim


def test_tables_with_several_variants(tmp_path: Path) -> None:
    a, b = small_model("net/v0000"), small_model("net/v0001")
    b.elements[0].yprim = 2 * Y_LINE
    ta, tb = io_parquet.to_tables(a, "sa"), io_parquet.to_tables(b, "sb")
    merged = {name: pa.concat_tables([ta[name], tb[name]]) for name in ta}
    assert io_parquet.from_tables(merged, "net/v0001", "sb") == b
    assert io_parquet.from_tables(merged, "net/v0000", "sa") == a


def test_tables_match_their_schemas() -> None:
    for name, table in io_parquet.to_tables(small_model(), state_id="s1").items():
        schema.check_schema(table, schema.TABLES[name], name)
    with pytest.raises(ValueError, match="does not match"):
        schema.check_schema(pa.table({"x": [1]}), schema.NODES, "variants/nodes")


def test_the_nominal_voltage_round_trips() -> None:
    model = small_model()
    model.nodes[1] = replace(model.nodes[1], kv_nominal_ln=0.22)
    back = io_parquet.from_tables(io_parquet.to_tables(model, "s1"), model.variant_id, "s1")
    assert [n.kv_nominal_ln for n in back.nodes] == [None, 0.22, None]
    assert back.nodes[1].nominal_v_ln == 220.0
    assert back.nodes[0].nominal_v_ln == back.nodes[0].kv_base_ln * 1e3


def test_ratings_are_stored_as_float32() -> None:
    model = small_model()
    model.elements[0].rating_a = 400.1
    back = io_parquet.from_tables(io_parquet.to_tables(model, "s1"), model.variant_id, "s1")
    assert back.elements[0].rating_a == float(np.float32(400.1))


def test_tables_are_referentially_consistent() -> None:
    from tests.integrity import check_variant_tables

    a, b = small_model("net/v0000"), small_model("net/v0001")
    ta, tb = io_parquet.to_tables(a, "sa"), io_parquet.to_tables(b, "sb")
    merged = {name: pa.concat_tables([ta[name], tb[name]]) for name in ta}
    assert check_variant_tables(merged) == []
    broken = dict(merged)
    terms = merged[io_parquet.TERMINALS]
    broken[io_parquet.TERMINALS] = terms.set_column(
        terms.schema.get_field_index("node_idx"),
        "node_idx",
        pa.array([7] * terms.num_rows, pa.int32()),
    )
    assert any("node_idx outside" in p for p in check_variant_tables(broken))
