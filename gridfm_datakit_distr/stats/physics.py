"""Physical statistics of a dataset, streamed part by part.

Per network, over the converged scenarios:

- ``vm_<phase>``: |V| of the phase nodes (pu of the line-to-neutral base), phases a, b, c and the
  split-phase halves s1, s2;
- ``neutral_v``: |V| of the neutral nodes to ground (pu of the line-to-neutral base);
- ``vuf``: |V2| / |V1| of every bus with phases a, b and c. The ratio is the same for
  phase-to-ground, phase-to-neutral and line-to-line voltages (a common-mode voltage has no
  sequence component; line voltages scale V1 and V2 by the same magnitude), so it is IEC's VUF;
- ``line_loading``: the largest |I| of a line's phase conductors (either end) over its normal
  ampacity; ``neutral_loading``: the same for its neutral conductors (currents are Yprim · V,
  recomputed from the stored primitive admittances at each scenario's network state);
- ``transformer_loading``: |S| into winding 1 over the rated power;
- violation counts per scenario: a phase node outside ``networks.voltage_limits``, a line or
  transformer above 1, a neutral conductor above its line's ampacity.

Distributions are fixed-bin histograms (quantiles to within one bin; exact count, minimum, maximum
and mean), so a dataset of any size is read in one pass with bounded memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.io_parquet import from_tables
from gridfm_datakit_distr.solve.extraction import ExtractionPlan, build_plan
from gridfm_datakit_distr.solve.pf_batch import MAX_CACHED_STATES
from gridfm_datakit_distr.views.datakit_layout import rows_by_key

ALPHA = np.exp(2j * np.pi / 3)
PHASE_LABELS = ("a", "b", "c", "s1", "s2")
RANGES = {
    "vm": (0.0, 2.0, 4000),
    "neutral_v": (0.0, 0.5, 5000),
    "vuf": (0.0, 0.2, 4000),
    "line_loading": (0.0, 5.0, 5000),
    "neutral_loading": (0.0, 5.0, 5000),
    "transformer_loading": (0.0, 5.0, 5000),
}
"""Histogram range and bins of each quantity: (low, high, bins); values outside are counted."""
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)
VARIANT_TABLES = (
    "variants/nodes",
    "variants/elements",
    "variants/element_terminals",
    "variants/yprim",
)
REPORTED = ("vm_a", "vm_b", "vm_c", "neutral_v", "vuf", "line_loading", "transformer_loading")


@dataclass(slots=True)
class Histogram:
    lo: float
    hi: float
    bins: int
    counts: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    """``bins`` counts, then below ``lo`` and above ``hi``."""
    n: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def __post_init__(self) -> None:
        if self.counts.size == 0:
            self.counts = np.zeros(self.bins + 2, dtype=np.int64)

    def add(self, values: NDArray[np.float64]) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if not v.size:
            return
        width = (self.hi - self.lo) / self.bins
        idx = np.floor((v - self.lo) / width).astype(np.int64)
        below, above = v < self.lo, v >= self.hi
        inside = ~(below | above)
        self.counts[: self.bins] += np.bincount(idx[inside], minlength=self.bins)
        self.counts[self.bins] += int(below.sum())
        self.counts[self.bins + 1] += int(above.sum())
        self.n += int(v.size)
        self.total += float(v.sum())
        self.minimum = min(self.minimum, float(v.min()))
        self.maximum = max(self.maximum, float(v.max()))

    def quantile(self, q: float) -> float:
        """The ``q`` quantile, linear within its bin (exact for values outside the range: the
        minimum or maximum)."""
        if not self.n:
            return float("nan")
        rank = q * self.n
        below = self.counts[self.bins]
        if rank <= below:
            return self.minimum
        cumulative = below + np.cumsum(self.counts[: self.bins])
        k = int(np.searchsorted(cumulative, rank))
        if k >= self.bins:
            return self.maximum
        width = (self.hi - self.lo) / self.bins
        before = cumulative[k - 1] if k > 0 else below
        inside = (rank - before) / max(self.counts[k], 1)
        return float(np.clip(self.lo + (k + inside) * width, self.minimum, self.maximum))

    def summary(self) -> dict[str, Any]:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "mean": self.total / self.n,
            "min": self.minimum,
            "max": self.maximum,
            **{f"p{round(q * 100):02d}": self.quantile(q) for q in QUANTILES},
        }


@dataclass(slots=True)
class NetworkStats:
    network_id: str
    scenarios: int = 0
    histograms: dict[str, Histogram] = field(default_factory=dict)
    violations: dict[str, int] = field(default_factory=lambda: {"voltage": 0})
    """Scenarios with a violation; ``thermal`` and ``neutral`` only where something is rated
    (a line ampacity declared by the source, a transformer's kVA)."""

    def histogram(self, name: str) -> Histogram:
        if name not in self.histograms:
            lo, hi, bins = RANGES[name.split("_", 1)[0] if name.startswith("vm_") else name]
            self.histograms[name] = Histogram(lo, hi, bins)
        return self.histograms[name]

    def summary(self) -> dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "violation_rates": {
                k: (v / self.scenarios if self.scenarios else None)
                for k, v in self.violations.items()
            },
            "distributions": {k: h.summary() for k, h in sorted(self.histograms.items())},
        }


@dataclass(slots=True)
class _Variant:
    """Index arrays of one variant's model and the tables to rebuild it at a network state."""

    variant_id: str
    base_v: NDArray[np.float64]
    conductor: NDArray[np.str_]
    phase: dict[str, NDArray[np.intp]]
    neutral: NDArray[np.intp]
    abc: NDArray[np.intp]
    """(buses, 3) node indices of the three-phase buses."""
    element_class: NDArray[np.str_]
    rating_a: NDArray[np.float64]
    rating_va: NDArray[np.float64]
    tables: dict[str, pa.Table]
    state_rows: dict[str, pa.Table] | None
    """Per state: its ``states/yprim_state`` rows (None: the dataset has no such table)."""
    plans: dict[str, tuple[ExtractionPlan, _Rows]] = field(default_factory=dict)
    """Per state: the extraction plan (Yprim blocks at the state) and its row layout; the
    most recent ``MAX_CACHED_STATES``."""


@dataclass(frozen=True, slots=True)
class _Rows:
    line_phase: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]
    """(rows of line phase conductors, group of their line, the line's ampacity per row)."""
    line_neutral: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]
    xf: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]
    """(rows of winding-1 positions, their node or -1, group of their transformer, its rating
    per group)."""


def _variant(root: Path, key: str, variant_id: str, yprim_state: pa.Table | None) -> _Variant:
    tables = {name: pq.read_table(root / name / f"part-{key}.parquet") for name in VARIANT_TABLES}
    state_rows = None
    if yprim_state is not None:  # cut once per variant: filtering per state is quadratic
        mine = yprim_state.filter(pc.equal(yprim_state["variant_id"], pa.scalar(variant_id)))
        state_rows = rows_by_key(mine, "state_id")
    nodes, elements = tables["variants/nodes"], tables["variants/elements"]
    conductor = np.array([str(c) for c in nodes.column("conductor").to_pylist()])
    bus = [str(b) for b in nodes.column("bus").to_pylist()]
    by_bus: dict[str, dict[str, int]] = {}
    for k, (b, c) in enumerate(zip(bus, conductor.tolist(), strict=True)):
        by_bus.setdefault(b, {})[c] = k
    abc = [[d["a"], d["b"], d["c"]] for d in by_bus.values() if {"a", "b", "c"} <= set(d)]
    order = np.argsort(elements.column("element_idx").to_numpy())
    return _Variant(
        variant_id=variant_id,
        base_v=nodes.column("kv_base_ln").to_numpy() * 1e3,
        conductor=conductor,
        phase={c: np.flatnonzero(conductor == c) for c in PHASE_LABELS},
        neutral=np.flatnonzero(conductor == "n"),
        abc=np.array(abc, dtype=np.intp).reshape(-1, 3),
        element_class=np.array(elements.column("element_class").to_pylist())[order],
        rating_a=np.array(elements.column("rating_a").to_pylist(), dtype=np.float64)[order],
        rating_va=np.array(elements.column("rating_va").to_pylist(), dtype=np.float64)[order],
        tables=tables,
        state_rows=state_rows,
    )


def _plan(m: _Variant, state_id: str) -> tuple[ExtractionPlan, _Rows]:
    """The extraction plan of the variant at a state (rebuilt from the tables) and its rows."""
    if state_id not in m.plans:
        tables = dict(m.tables)
        if m.state_rows is not None:
            empty = schema.YPRIM_STATE.empty_table()
            tables["states/yprim_state"] = m.state_rows.get(state_id, empty)
        if len(m.plans) >= MAX_CACHED_STATES:
            m.plans.pop(next(iter(m.plans)))  # the oldest
        model = from_tables(tables, m.variant_id, state_id)
        plan = build_plan(model)
        select = plan.select.tocsr()
        node = np.full(plan.element_idx.size, -1, dtype=np.intp)
        filled = np.diff(select.indptr) > 0
        node[filled] = select.indices
        m.plans[state_id] = (plan, _rows(m, plan.element_idx, plan.terminal, node))
    return m.plans[state_id]


def _rows(
    m: _Variant,
    element: NDArray[np.int32],
    terminal: NDArray[np.int8],
    node: NDArray[np.intp],
) -> _Rows:
    """Row layout of the element currents of the extraction plan (element, terminal, node)."""
    conductor = np.where(node >= 0, m.conductor[np.maximum(node, 0)], "0")
    cls, amps, va = m.element_class[element], m.rating_a[element], m.rating_va[element]
    line = np.isin(cls, ("line", "switch")) & np.isfinite(amps) & (amps > 0)
    ph = line & np.isin(conductor, PHASE_LABELS)
    nn = line & (conductor == "n")
    xf = (cls == "transformer") & (terminal == 0) & np.isfinite(va) & (va > 0)

    def grouped(mask: NDArray[np.bool_]) -> NDArray[np.intp]:
        return np.unique(element[mask], return_inverse=True)[1].astype(np.intp)

    xf_groups = grouped(xf)
    n_xf = int(xf_groups.max(initial=-1)) + 1
    ratings = np.array([va[xf][xf_groups == g][0] for g in range(n_xf)])
    return _Rows(
        line_phase=(np.flatnonzero(ph), grouped(ph), amps[ph]),
        line_neutral=(np.flatnonzero(nn), grouped(nn), amps[nn]),
        xf=(np.flatnonzero(xf), node[xf], xf_groups, ratings),
    )


def _reduce(values: NDArray[Any], groups: NDArray[np.intp], op: np.ufunc) -> NDArray[Any]:
    """``op`` over the columns of each group, per row: (scenarios, columns) -> (scenarios,
    groups); groups are numbered 0 .. n - 1."""
    order = np.argsort(groups, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(groups[order]) != 0])
    return np.asarray(op.reduceat(values[:, order], starts, axis=1))


def physics(root: Path, vmin: float, vmax: float) -> dict[str, NetworkStats]:
    """Statistics of every network of the dataset in ``root`` (converged scenarios). Element
    currents are recomputed as Yprim · V from the stored tables (they need not be stored)."""
    networks = {
        int(r["network_idx"]): str(r["network_id"])
        for r in pq.read_table(root / "catalog" / "networks.parquet").to_pylist()
    }
    single = root / "states" / "yprim_state.parquet"
    yprim_state = pq.read_table(single) if single.exists() else None
    stats: dict[str, NetworkStats] = {}
    current: _Variant | None = None
    for part in sorted((root / "scenarios" / "node_results").glob("part-*.parquet")):
        key = part.name[len("part-") : -len(".parquet")]
        i, k, _ = key.split("-")
        meta = pq.read_table(root / "scenarios" / "scenario_meta" / part.name)
        state_of = dict(
            zip(
                meta.column("scenario_id").to_pylist(),
                meta.column("state_id").to_pylist(),
                strict=True,
            )
        )
        variant_id = str(meta.column("variant_id")[0].as_py())
        if current is None or current.variant_id != variant_id:  # one variant at a time
            current = _variant(root, f"{i}-{k}", variant_id, yprim_state)
        s = stats.setdefault(networks[int(i)], NetworkStats(networks[int(i)]))
        _add_chunk(s, current, part, state_of, vmin, vmax)
    return stats


def _add_chunk(
    s: NetworkStats,
    m: _Variant,
    nodes_part: Path,
    state_of: dict[Any, Any],
    vmin: float,
    vmax: float,
) -> None:
    nodes = pq.read_table(nodes_part)
    n_nodes = m.base_v.size
    if nodes.num_rows == 0:
        return
    if nodes.num_rows % n_nodes:
        raise ValueError(f"{nodes_part}: {nodes.num_rows} rows for {n_nodes} nodes")
    n_scen = nodes.num_rows // n_nodes
    idx = nodes.column("node_idx").to_numpy().reshape(n_scen, n_nodes)
    if not (idx == np.arange(n_nodes)).all():
        raise ValueError(f"{nodes_part}: rows are not in node order per scenario")
    v = nodes.column("v_re").to_numpy() + 1j * nodes.column("v_im").to_numpy()
    v = v.reshape(n_scen, n_nodes)
    sids = nodes.column("scenario_id").to_numpy().reshape(n_scen, n_nodes)[:, 0]
    vm = np.abs(v) / m.base_v
    s.scenarios += n_scen
    out_of_band = np.zeros(n_scen, dtype=bool)
    for c, cols in m.phase.items():
        if cols.size:
            s.histogram(f"vm_{c}").add(vm[:, cols])
            out_of_band |= ((vm[:, cols] < vmin) | (vm[:, cols] > vmax)).any(axis=1)
    s.violations["voltage"] += int(out_of_band.sum())
    if m.neutral.size:
        s.histogram("neutral_v").add(vm[:, m.neutral])
    if m.abc.size:
        vabc = v[:, m.abc]  # (scenarios, buses, 3)
        v1 = vabc[..., 0] + ALPHA * vabc[..., 1] + ALPHA**2 * vabc[..., 2]
        v2 = vabc[..., 0] + ALPHA**2 * vabc[..., 1] + ALPHA * vabc[..., 2]
        s.histogram("vuf").add(np.abs(v2) / np.abs(v1))
    thermal = np.zeros(n_scen, dtype=bool)
    rated = False
    states = np.array([str(state_of[int(sid)]) for sid in sids])
    for state in np.unique(states).tolist():
        mine = np.flatnonzero(states == state)
        plan, layout = _plan(m, state)
        i_local = np.asarray((plan.yblk @ (plan.select @ v[mine].T)).T)  # Yprim · V, (scen, rows)
        found = _loadings(s, layout, v[mine], i_local)
        if found is not None:
            thermal[mine] |= found
            rated = True
    if rated:
        s.violations["thermal"] = s.violations.get("thermal", 0) + int(thermal.sum())


def _loadings(
    s: NetworkStats, layout: _Rows, v: NDArray[np.complex128], i_local: NDArray[np.complex128]
) -> NDArray[np.bool_] | None:
    """Add the line, neutral and transformer loadings; returns the thermal violation per row,
    None when no line or transformer is rated."""
    if not (layout.line_phase[0].size or layout.xf[0].size):
        return None
    thermal = np.zeros(v.shape[0], dtype=bool)
    i_abs = np.abs(i_local)
    rows, line, amps = layout.line_phase
    if rows.size:
        loading = _reduce(i_abs[:, rows] / amps, line, np.maximum)
        s.histogram("line_loading").add(loading)
        thermal |= (loading > 1).any(axis=1)
    rows, line, amps = layout.line_neutral
    if rows.size:
        loading = _reduce(i_abs[:, rows] / amps, line, np.maximum)
        s.histogram("neutral_loading").add(loading)
        s.violations["neutral"] = s.violations.get("neutral", 0) + int(
            (loading > 1).any(axis=1).sum()
        )
    rows, node, xf, rating = layout.xf
    if rows.size:
        v_xf = np.where(node >= 0, v[:, np.maximum(node, 0)], 0)
        power = _reduce(v_xf * np.conj(i_local[:, rows]), xf, np.add)
        loading = np.abs(power) / rating
        s.histogram("transformer_loading").add(loading)
        thermal |= (loading > 1).any(axis=1)
    return thermal


def write_report(root: Path, stats: dict[str, NetworkStats], vmin: float, vmax: float) -> Path:
    """``stats/stats.json`` (per network: violation rates, distribution summaries and the
    histograms' counts); returns its path."""
    out = root / "stats"
    out.mkdir(exist_ok=True)
    data = {
        "voltage_limits_pu": [vmin, vmax],
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "networks": {
            nid: {
                **s.summary(),
                "histograms": {k: h.counts.tolist() for k, h in s.histograms.items()},
            }
            for nid, s in sorted(stats.items())
        },
    }
    path = out / "stats.json"
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def report_text(stats: dict[str, NetworkStats]) -> str:
    """A short text report: violation rates and the main quantiles per network."""
    lines = []
    for nid, s in sorted(stats.items()):
        rates = ", ".join(
            f"{k} {v / s.scenarios:.1%}" for k, v in s.violations.items() if s.scenarios
        )
        lines.append(f"{nid}: {s.scenarios} converged scenarios; violations: {rates}")
        for name in REPORTED:
            h = s.histograms.get(name)
            if h is not None and h.n:
                p = h.summary()
                lines.append(
                    f"  {name}: p01 {p['p01']:.4f}, p50 {p['p50']:.4f}, p99 {p['p99']:.4f}, "
                    f"max {p['max']:.4f}"
                )
    return "\n".join(lines) + "\n"
