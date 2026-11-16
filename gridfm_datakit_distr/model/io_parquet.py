"""Parquet (de)serialisation of the electrical model (outputs.md §4 and §5).

A model is split into the variant-level tables (nodes, elements, element terminals, Yprim of
tap-independent PD elements) and, if it has tap-dependent elements, the state-level table of
their Yprims (``states/yprim_state``, keyed by ``state_id``). Yprims are stored as their
non-zero entries; ratings are stored as float32 (outputs.md §4), so they round-trip exactly only
when representable in float32. When reading, an enabled PD element without stored entries gets
an all-zero Yprim; PC elements, disabled PD elements and tap-dependent elements read without
their state get ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import ComplexMatrix, ElectricalModel, Element, Node
from gridfm_datakit_distr.model.ids import split_variant_id

NODES = "variants/nodes"
ELEMENTS = "variants/elements"
TERMINALS = "variants/element_terminals"
YPRIM = "variants/yprim"
YPRIM_STATE = "states/yprim_state"


def _nodes_table(model: ElectricalModel) -> pa.Table:
    nodes = model.nodes
    return pa.table(
        {
            "variant_id": [model.variant_id] * len(nodes),
            "node_idx": [n.idx for n in nodes],
            "bus": [n.bus for n in nodes],
            "conductor": [n.conductor for n in nodes],
            "opendss_node": [n.opendss_node for n in nodes],
            "kv_base_ln": [n.kv_base_ln for n in nodes],
            "kv_nominal_ln": [n.kv_nominal_ln for n in nodes],
            "is_grounded": [n.is_grounded for n in nodes],
            "r_ground_ohm": [n.r_ground_ohm for n in nodes],
            "x_ground_ohm": [n.x_ground_ohm for n in nodes],
            "is_source": [n.is_source for n in nodes],
            "backbone_bus": [None] * len(nodes),
        },
        schema=schema.NODES,
    )


def _elements_table(model: ElectricalModel) -> pa.Table:
    els = model.elements
    return pa.table(
        {
            "variant_id": [model.variant_id] * len(els),
            "element_idx": [e.idx for e in els],
            "name": [e.name for e in els],
            "element_class": [e.element_class for e in els],
            "is_pd": [e.is_pd for e in els],
            "n_terminals": [e.n_terminals for e in els],
            "n_conductors": [e.n_conductors for e in els],
            "enabled": [e.enabled for e in els],
            "is_tap_dependent": [e.is_tap_dependent for e in els],
            "rating_a": [e.rating_a for e in els],
            "rating_va": [e.rating_va for e in els],
        },
        schema=schema.ELEMENTS,
    )


def _terminals_table(model: ElectricalModel) -> pa.Table:
    element_idx: list[int] = []
    terminal: list[int] = []
    position: list[int] = []
    node_idx: list[int] = []
    for e in model.elements:
        for t, nodes in enumerate(e.terminals):
            for p, node in enumerate(nodes):
                element_idx.append(e.idx)
                terminal.append(t)
                position.append(p)
                node_idx.append(node)
    return pa.table(
        {
            "variant_id": [model.variant_id] * len(node_idx),
            "element_idx": element_idx,
            "terminal": terminal,
            "position": position,
            "node_idx": node_idx,
        },
        schema=schema.ELEMENT_TERMINALS,
    )


def _yprim_columns(variant_id: str, elements: list[Element]) -> dict[str, np.ndarray | list[str]]:
    element_idx, rows, cols, values = [], [], [], []
    for e in elements:
        if e.yprim is None:
            continue
        r, c = np.nonzero(e.yprim)
        element_idx.append(np.full(r.size, e.idx, dtype=np.int32))
        rows.append(r.astype(np.int16))
        cols.append(c.astype(np.int16))
        values.append(e.yprim[r, c])
    v = np.concatenate(values) if values else np.zeros(0, dtype=np.complex128)
    n = v.size
    return {
        "variant_id": [variant_id] * n,
        "element_idx": np.concatenate(element_idx) if element_idx else np.zeros(0, np.int32),
        "row": np.concatenate(rows) if rows else np.zeros(0, np.int16),
        "col": np.concatenate(cols) if cols else np.zeros(0, np.int16),
        "g": v.real.astype(np.float64),
        "b": v.imag.astype(np.float64),
    }


def variant_level_tables(model: ElectricalModel) -> dict[str, pa.Table]:
    """Nodes, elements, terminals and the tap-independent Yprims of ``model``; the tap-dependent
    Yprims belong to network states (:func:`to_tables` with a ``state_id``)."""
    pd = [e for e in model.elements if e.is_pd]
    return {
        NODES: _nodes_table(model),
        ELEMENTS: _elements_table(model),
        TERMINALS: _terminals_table(model),
        YPRIM: pa.table(
            _yprim_columns(model.variant_id, [e for e in pd if not e.is_tap_dependent]),
            schema=schema.YPRIM,
        ),
    }


def to_tables(model: ElectricalModel, state_id: str | None = None) -> dict[str, pa.Table]:
    """Variant-level tables of ``model``, plus ``states/yprim_state`` if it has tap-dependent
    elements (then ``state_id`` is required)."""
    tables = variant_level_tables(model)
    dependent = [e for e in model.elements if e.is_pd and e.is_tap_dependent]
    if dependent:
        if state_id is None:
            raise ValueError("the model has tap-dependent elements: a state_id is required")
        columns = _yprim_columns(model.variant_id, dependent)
        tables[YPRIM_STATE] = pa.table(
            {"state_id": [state_id] * len(columns["g"]), **columns}, schema=schema.YPRIM_STATE
        )
    return tables


def _select(table: pa.Table, column: str, value: str) -> pa.Table:
    return table.filter(pc.equal(table[column], pa.scalar(value)))


def _yprims(table: pa.Table, sizes: Mapping[int, int]) -> dict[int, ComplexMatrix]:
    out: dict[int, ComplexMatrix] = {}
    idx = table["element_idx"].to_numpy()
    rows = table["row"].to_numpy()
    cols = table["col"].to_numpy()
    values = table["g"].to_numpy() + 1j * table["b"].to_numpy()
    for element in np.unique(idx):
        k = int(element)
        sel = idx == element
        matrix = np.zeros((sizes[k], sizes[k]), dtype=np.complex128)
        matrix[rows[sel], cols[sel]] = values[sel]
        out[k] = matrix
    return out


def from_tables(
    tables: Mapping[str, pa.Table], variant_id: str, state_id: str | None = None
) -> ElectricalModel:
    """Rebuild the model of ``variant_id`` (and ``state_id`` for tap-dependent Yprims)."""
    for name, table in tables.items():
        schema.check_schema(table, schema.TABLES[name], name)
    nodes_t = _select(tables[NODES], "variant_id", variant_id).sort_by("node_idx")
    elements_t = _select(tables[ELEMENTS], "variant_id", variant_id).sort_by("element_idx")
    terms_t = _select(tables[TERMINALS], "variant_id", variant_id).sort_by(
        [("element_idx", "ascending"), ("terminal", "ascending"), ("position", "ascending")]
    )
    nodes = [
        Node(
            idx=row["node_idx"],
            bus=row["bus"],
            conductor=row["conductor"],
            opendss_node=row["opendss_node"],
            kv_base_ln=row["kv_base_ln"],
            is_grounded=row["is_grounded"],
            r_ground_ohm=row["r_ground_ohm"],
            x_ground_ohm=row["x_ground_ohm"],
            is_source=row["is_source"],
            kv_nominal_ln=row["kv_nominal_ln"],
        )
        for row in nodes_t.to_pylist()
    ]
    terminals: dict[int, list[list[int]]] = {}
    for row in terms_t.to_pylist():
        per_element = terminals.setdefault(row["element_idx"], [])
        if row["terminal"] == len(per_element):
            per_element.append([])
        per_element[row["terminal"]].append(row["node_idx"])
    rows = elements_t.to_pylist()
    sizes = {r["element_idx"]: r["n_terminals"] * r["n_conductors"] for r in rows}
    yprims = _yprims(_select(tables[YPRIM], "variant_id", variant_id), sizes)
    if YPRIM_STATE in tables:
        if state_id is None:
            raise ValueError("state-level Yprims given without a state_id")
        state = _select(tables[YPRIM_STATE], "state_id", state_id)
        state = _select(state, "variant_id", variant_id)
        yprims.update(_yprims(state, sizes))

    def yprim_of(r: dict[str, Any]) -> ComplexMatrix | None:
        k = r["element_idx"]
        if k in yprims:
            return yprims[k]
        if r["is_pd"] and r["enabled"] and not r["is_tap_dependent"]:
            return np.zeros((sizes[k], sizes[k]), dtype=np.complex128)  # all entries were zero
        return None  # PC element, disabled PD element, or state not given

    elements = [
        Element(
            idx=r["element_idx"],
            name=r["name"],
            element_class=r["element_class"],
            is_pd=r["is_pd"],
            enabled=r["enabled"],
            terminals=terminals[r["element_idx"]],
            yprim=yprim_of(r),
            is_tap_dependent=r["is_tap_dependent"],
            rating_a=r["rating_a"],
            rating_va=r["rating_va"],
        )
        for r in rows
    ]
    network_id, _ = split_variant_id(variant_id)
    return ElectricalModel(
        network_id=network_id, variant_id=variant_id, nodes=nodes, elements=elements
    )
