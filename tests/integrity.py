"""Referential integrity of the variant- and state-level tables (pattern from gridfm-datakit tests).

Checks that indices are dense and equal to row positions, and that every foreign key exists in
its parent table. Returns the list of problems (empty when the tables are consistent).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

NODES = "variants/nodes"
ELEMENTS = "variants/elements"
TERMINALS = "variants/element_terminals"
YPRIM = "variants/yprim"
YPRIM_STATE = "states/yprim_state"


def _dense(values: np.ndarray, what: str) -> list[str]:
    if values.size and not np.array_equal(np.sort(values), np.arange(values.size)):
        return [f"{what}: indices are not 0..{values.size - 1} without gaps or duplicates"]
    return []


def _rows(tables: Mapping[str, pa.Table], name: str, variant: str) -> pa.Table:
    table = tables[name]
    return table.filter(pc.equal(table["variant_id"], pa.scalar(variant)))


def check_variant_tables(tables: Mapping[str, pa.Table]) -> list[str]:
    """Integrity problems of the tables of every variant they contain."""
    problems: list[str] = []
    variants = pc.unique(tables[NODES]["variant_id"]).to_pylist()
    for variant in variants:
        nodes, elements, terms = (
            _rows(tables, n, str(variant)) for n in (NODES, ELEMENTS, TERMINALS)
        )
        n_nodes = nodes.num_rows
        problems += _dense(nodes["node_idx"].to_numpy(), f"{variant} nodes")
        problems += _dense(elements["element_idx"].to_numpy(), f"{variant} elements")
        size = {
            int(e): int(t) * int(c)
            for e, t, c in zip(
                elements["element_idx"].to_numpy(),
                elements["n_terminals"].to_numpy(),
                elements["n_conductors"].to_numpy(),
                strict=True,
            )
        }
        node_idx = terms["node_idx"].to_numpy()
        if node_idx.size and (node_idx.min() < -1 or node_idx.max() >= n_nodes):
            problems.append(f"{variant} element_terminals: node_idx outside [-1, {n_nodes})")
        per_element = np.bincount(terms["element_idx"].to_numpy(), minlength=len(size))
        wrong = [e for e, n in size.items() if per_element[e] != n]
        if wrong:
            problems.append(
                f"{variant} element_terminals: wrong row count for elements {wrong[:5]}"
            )
        is_pd = dict(
            zip(elements["element_idx"].to_pylist(), elements["is_pd"].to_pylist(), strict=True)
        )
        names = [YPRIM] + ([YPRIM_STATE] if YPRIM_STATE in tables else [])
        for name in names:
            y = _rows(tables, name, str(variant))
            for e, r, c in zip(
                y["element_idx"].to_numpy(), y["row"].to_numpy(), y["col"].to_numpy(), strict=True
            ):
                k = int(e)
                if not is_pd.get(k, False):
                    problems.append(f"{variant} {name}: element {k} is not a PD element")
                    break
                if not (0 <= r < size[k] and 0 <= c < size[k]):
                    problems.append(f"{variant} {name}: entry ({r}, {c}) outside element {k}")
                    break
    return problems
