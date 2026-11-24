"""Reference loader: a dataset's tables as the arrays of a power flow, without OpenDSS.

Per variant (:class:`VariantData`): the model rebuilt from its tables, its network admittance as
compiled, the source (ideal voltages behind an admittance) and, for every load and DER element, the
branches it draws its power through. Per scenario (:class:`ScenarioData`): the network admittance
of its state (the variant's with the state's changed entries), the source voltage, the setpoints
and the stored node voltages.
:mod:`gridfm_datakit_distr.validate.closed_loop` solves the power flow from these alone.
Nothing here imports the engine (a test checks it).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from numpy.typing import NDArray

from gridfm_datakit_distr.extract.source import scenario_e
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.writer import part_key, read_states
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.ground import unit_branches
from gridfm_datakit_distr.model.ids import split_scenario_id
from gridfm_datakit_distr.model.io_parquet import (
    ELEMENTS,
    NODES,
    TERMINALS,
    YPRIM,
    YPRIM_STATE,
    from_tables,
)

VARIANT_TABLES = (NODES, ELEMENTS, TERMINALS, YPRIM)
"""The tables of a variant's model (Yprims of the tap-independent elements)."""
UNIT_CLASSES = ("load", "generator")
META_COLUMNS = (
    "scenario_id", "variant_id", "state_id", "pf_status", "source_v_pu", "source_u2",
    "source_u2_angle_rad",
)  # fmt: skip
"""The columns of ``scenarios/scenario_meta`` the reader needs."""


@dataclass(frozen=True, slots=True, eq=False)
class VariantData:
    variant_id: str
    model: ElectricalModel
    """Nodes, elements and tap-independent Yprims (tap-dependent ones are in each state's Y)."""
    base_v: NDArray[np.float64]
    """Line-to-neutral base voltage of each node (V)."""
    y_base: sp.csr_matrix
    """Network admittance (S) of the variant as compiled (``variants/ybus``)."""
    source_nodes: NDArray[np.int64]
    y_s: NDArray[np.complex128]
    """Source admittance (S) between its ideal voltages and :attr:`source_nodes`."""
    e_v: NDArray[np.complex128]
    """Ideal source voltages (V) at 1 pu."""
    branches: dict[int, list[tuple[int, int]]]
    """Per load or generator element: the node pairs it draws its power through, an equal share
    each (second node may be ground, -1)."""


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioData:
    scenario_id: int
    variant: VariantData
    state_id: str | None
    pf_status: str
    source_v_pu: float | None
    source_u2: float | None
    """Negative-sequence ratio of the source (0 or null: balanced)."""
    source_u2_angle_rad: float | None
    y_state: sp.csr_matrix | None
    """Network admittance (S) of the scenario's state; None without a state."""
    s_drawn: dict[int, complex]
    """Per load or DER element: the complex power it draws at its setpoint (VA; DER negative)."""
    v: NDArray[np.complex128] | None
    """Stored node voltages (V); None if the scenario has no power-flow label."""

    @property
    def source_e(self) -> NDArray[np.complex128] | None:
        """The scenario's ideal source voltages (V), :func:`scenario_e`; None without draws."""
        if self.source_v_pu is None:
            return None
        u2 = self.source_u2 or 0.0
        return scenario_e(self.variant.e_v, self.source_v_pu, u2, self.source_u2_angle_rad)


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


class VariantTables:
    """A variant's model rebuilt from its stored tables, at any of its network states."""

    def __init__(
        self, root: Path, key: str, variant_id: str, yprim_state: pa.Table | None = None
    ) -> None:
        """``key``: the variant's part key (:func:`part_key`); ``yprim_state``: the dataset's
        ``states/yprim_state`` (:func:`read_states`), None to rebuild without states."""
        self.variant_id = variant_id
        self.tables = {
            name: pq.read_table(root / name / f"part-{key}.parquet") for name in VARIANT_TABLES
        }
        self._state_rows: dict[str, pa.Table] | None = None
        if yprim_state is not None:  # cut once per variant: filtering per state is quadratic
            mine = yprim_state.filter(pc.equal(yprim_state["variant_id"], pa.scalar(variant_id)))
            self._state_rows = rows_by_key(mine, "state_id")

    def model(self, state_id: str | None = None) -> ElectricalModel:
        """The model at ``state_id``, with the tap-dependent Yprims of that state; without a state
        (or without ``yprim_state``) the tap-dependent elements have none."""
        tables = dict(self.tables)
        if state_id is not None and self._state_rows is not None:
            empty = schema.YPRIM_STATE.empty_table()
            tables[YPRIM_STATE] = self._state_rows.get(state_id, empty)
        return from_tables(tables, self.variant_id, state_id)


def admittance(table: pa.Table, n: int) -> sp.csr_matrix:
    """The ``n`` x ``n`` matrix of the ``row``, ``col``, ``g``, ``b`` rows of ``table`` (S)."""
    values = table.column("g").to_numpy() + 1j * table.column("b").to_numpy()
    rows, cols = table.column("row").to_numpy(), table.column("col").to_numpy()
    return sp.csr_matrix((values, (rows, cols)), shape=(n, n))


def with_changes(base: sp.csr_matrix, changes: pa.Table) -> sp.csr_matrix:
    """``base`` with the entries of ``changes`` (``row``, ``col``, ``g``, ``b``) in place of its
    own; a changed entry that is zero is dropped: a state's admittance from its variant's."""
    old = base.tocoo()
    rows = changes.column("row").to_numpy().astype(np.int64)
    cols = changes.column("col").to_numpy().astype(np.int64)
    n = base.shape[1]
    keep = ~np.isin(old.row.astype(np.int64) * n + old.col, rows * n + cols)
    values = changes.column("g").to_numpy() + 1j * changes.column("b").to_numpy()
    y = sp.csr_matrix(
        (
            np.concatenate([old.data[keep], values]),
            (np.concatenate([old.row[keep], rows]), np.concatenate([old.col[keep], cols])),
        ),
        shape=base.shape,
    )
    y.eliminate_zeros()
    return y


def _read(path: Path, filters: Any = None, columns: list[str] | None = None) -> pa.Table:
    """A table stored as one file, as parts, or both (before finalisation)."""
    parts = []
    if path.with_suffix(".parquet").exists():
        parts.append(pq.read_table(path.with_suffix(".parquet"), filters=filters, columns=columns))
    if path.is_dir():
        parts.append(pq.read_table(path, filters=filters, columns=columns))
    if not parts:
        raise FileNotFoundError(path)
    return pa.concat_tables(parts)


class DatasetReader:
    """Read a generated dataset; variants and states are cached."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.meta = _read(
            root / "scenarios" / "scenario_meta",
            columns=[*META_COLUMNS],
        )
        self._variants: dict[str, VariantData] = {}
        self._states: dict[str, sp.csr_matrix] = {}

    def variant(self, variant_id: str) -> VariantData:
        if variant_id not in self._variants:
            self._variants[variant_id] = self._variant(variant_id)
        return self._variants[variant_id]

    def _variant(self, variant_id: str) -> VariantData:
        where = [("variant_id", "=", variant_id)]
        mine = self.meta.filter(pc.equal(self.meta["variant_id"], pa.scalar(variant_id)))
        i, k, _ = split_scenario_id(int(mine.column("scenario_id")[0].as_py()))
        key = part_key(i, k)
        model = VariantTables(self.root, key, variant_id).model()
        y_base = pq.read_table(self.root / "variants" / "ybus" / f"part-{key}.parquet")
        source = _read(self.root / "variants" / "source", where).sort_by("position")
        y_rows = _read(self.root / "variants" / "source_yprim", where)
        position = {p: k for k, p in enumerate(source.column("position").to_pylist())}
        n = len(position)
        y_s = np.zeros((n, n), dtype=np.complex128)
        for r in y_rows.to_pylist():
            y_s[position[r["row"]], position[r["col"]]] = complex(r["g"], r["b"])
        branches = {}
        for e in model.elements:
            if e.enabled and e.element_class in UNIT_CLASSES:
                found = unit_branches(model, e)
                if found is None:
                    raise ValueError(f"{variant_id}: {e.name} has a connection the reader omits")
                branches[e.idx] = found
        return VariantData(
            variant_id=variant_id,
            model=model,
            base_v=np.array([x.kv_base_ln for x in model.nodes]) * 1e3,
            y_base=admittance(y_base, model.n_nodes),
            source_nodes=np.asarray(source.column("node_idx").to_numpy(), dtype=np.int64),
            y_s=y_s,
            e_v=source.column("e_re").to_numpy() + 1j * source.column("e_im").to_numpy(),
            branches=branches,
        )

    def _state(self, state_id: str, variant: VariantData) -> sp.csr_matrix:
        if state_id not in self._states:
            t = read_states(self.root, "states/ybus", [("state_id", "=", state_id)])
            if t is None:
                raise FileNotFoundError(self.root / "states" / "ybus")
            self._states[state_id] = with_changes(variant.y_base, t)
        return self._states[state_id]

    def scenarios(self, ids: Sequence[int]) -> Iterator[ScenarioData]:
        """The scenarios ``ids`` (in that order)."""
        where = [("scenario_id", "in", [int(i) for i in ids])]
        meta = {r["scenario_id"]: r for r in self.meta.filter(_mask(self.meta, ids)).to_pylist()}
        drawn: dict[int, dict[int, complex]] = {}
        for name, sign in (("loads", 1.0), ("der", -1.0)):
            t = _read(
                self.root / "scenarios" / name,
                where,
                ["scenario_id", "element_idx", "p_w", "q_var"],
            )
            for r in t.to_pylist():
                s = sign * complex(r["p_w"], r["q_var"])
                drawn.setdefault(r["scenario_id"], {})[r["element_idx"]] = s
        nodes = _read(self.root / "scenarios" / "node_results", where).sort_by(
            [("scenario_id", "ascending"), ("node_idx", "ascending")]
        )
        stored: dict[int, NDArray[np.complex128]] = {}
        sids = nodes.column("scenario_id").to_numpy()
        v_all = nodes.column("v_re").to_numpy() + 1j * nodes.column("v_im").to_numpy()
        for sid in np.unique(sids):
            stored[int(sid)] = v_all[sids == sid]
        for sid in ids:
            m = meta[int(sid)]
            variant = self.variant(m["variant_id"])
            state = m["state_id"]
            yield ScenarioData(
                scenario_id=int(sid),
                variant=variant,
                state_id=state,
                pf_status=m["pf_status"],
                source_v_pu=m["source_v_pu"],
                source_u2=m["source_u2"],
                source_u2_angle_rad=m["source_u2_angle_rad"],
                y_state=None if state is None else self._state(state, variant),
                s_drawn=drawn.get(int(sid), {}),
                v=stored.get(int(sid)),
            )


def _mask(table: pa.Table, ids: Sequence[int]) -> pa.Array[Any]:
    return pc.is_in(table.column("scenario_id"), pa.array([int(i) for i in ids], pa.int64()))
