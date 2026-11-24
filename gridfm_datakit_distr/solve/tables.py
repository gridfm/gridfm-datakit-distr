"""Tables of a variant and of a chunk of its scenarios (outputs.md §4-§6).

``variant_tables``: the variant's row and its model (nodes, elements, terminals, tap-independent
Yprims). ``variant_ybus_table``: the variant's network admittance as compiled. ``state_tables``: the
states a chunk met (row, tap-dependent Yprims, the entries of their network admittance that differ
from the variant's).
``chunk_tables``: ``scenario_meta`` for every scenario; ``loads`` and ``der`` (setpoints) for every
scenario with draws; ``node_results`` (and, when asked, ``element_results``) for converged
scenarios only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import numpy as np
import pyarrow as pa
import scipy.sparse as sp

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.writer import STATES_PART
from gridfm_datakit_distr.model import io_parquet
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.state import NetworkState
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner, der_setpoints
from gridfm_datakit_distr.variants.generate import VariantResult, variants_table


def variant_tables(result: VariantResult) -> dict[str, pa.Table]:
    """``variants/variants`` and the variant's model tables."""
    if result.model is None:
        raise ValueError(f"{result.variant_id} failed: no tables")
    tables = {"variants/variants": variants_table([result])}
    tables.update(io_parquet.variant_level_tables(result.model))  # tap-dependent: per state
    return tables


def variant_ybus_table(variant_id: str, y: sp.csr_matrix) -> pa.Table:
    """``variants/ybus``: the non-zero entries of ``y``, the variant's network admittance."""
    coo = y.tocoo()
    return pa.table(
        {
            "variant_id": [variant_id] * coo.nnz,
            "row": coo.row.astype(np.int32),
            "col": coo.col.astype(np.int32),
            "g": coo.data.real,
            "b": coo.data.imag,
        },
        schema=schema.VARIANT_YBUS,
    )


def ybus_changes(
    y: sp.csr_matrix, base: sp.csr_matrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, cols, values)`` of the entries of ``y`` that differ from ``base`` (an absent
    entry is zero, so a vanished one comes back as a zero), in row-major order."""
    changed = cast(sp.csr_matrix, y != base).tocoo()  # elementwise: a boolean sparse matrix
    order = np.lexsort((changed.col, changed.row))
    rows, cols = changed.row[order], changed.col[order]
    if not rows.size:  # scipy returns a sparse matrix for an empty selection
        return rows, cols, np.zeros(0, dtype=np.complex128)
    return rows, cols, np.asarray(y[rows, cols], dtype=np.complex128).ravel()


def state_tables(
    states: dict[str, tuple[NetworkState, ElectricalModel]], base: sp.csr_matrix
) -> dict[str, pa.Table]:
    """Rows of the states met by a chunk (``states/states`` without the count); ``base``: the
    variant's network admittance (:func:`variant_ybus_table`)."""
    if not states:
        return {}
    rows, yprims, ybus = [], [], []
    for sid, (state, model) in states.items():
        rows.append(
            {
                "state_id": sid,
                "variant_id": state.variant_id,
                "taps": state.taps_json(),
                "capacitor_states": state.capacitors_json(),
            }
        )
        yprim = io_parquet.state_table(model, sid)
        if yprim is not None:
            yprims.append(yprim)
        rows_y, cols_y, values = ybus_changes(model.ybus(), base)
        ybus.append(
            pa.table(
                {
                    "state_id": [sid] * values.size,
                    "row": rows_y.astype(np.int32),
                    "col": cols_y.astype(np.int32),
                    "g": values.real,
                    "b": values.imag,
                },
                schema=schema.YBUS_STATE,
            )
        )
    out = {
        "states/states": pa.Table.from_pylist(rows, schema=STATES_PART),
        "states/ybus": pa.concat_tables(ybus),
    }
    if yprims:
        out["states/yprim_state"] = pa.concat_tables(yprims)
    return out


def chunk_tables(
    runner: VariantRunner,
    outcomes: Sequence[ScenarioOutcome],
    *,
    precision: schema.Precision,
    config_hash: str,
    versions: dict[str, str],
) -> dict[str, pa.Table]:
    """The scenario-level tables of a chunk; ``element_results`` only with the runner's
    ``labels.element_currents`` (otherwise they are Yprim · V)."""
    element_currents = runner.config.labels.element_currents
    index = {e.name: e.idx for e in runner.model.elements}
    load_idx = np.array([index[c.element] for c in runner.customers], dtype=np.int32)
    der_idx = np.array([index[g] for g in runner.der_generators], dtype=np.int32)
    pv_units = runner.pv.units if runner.pv is not None else ()
    bess_units = runner.storage.units if runner.storage is not None else ()
    dg_units = runner.dg.units if runner.dg is not None else ()
    der_type = ["pv"] * len(pv_units) + ["bess"] * len(bess_units) + ["dg"] * len(dg_units)
    dg_max = np.array([1e3 * u.kw for u in dg_units])
    rated = np.r_[[u.s_rated_va for u in pv_units], [1e3 * u.kw for u in bess_units], dg_max]
    engines = json.dumps({k: v for k, v in versions.items() if k != "generator"}, sort_keys=True)
    meta = [
        {
            "scenario_id": o.scenario_id,
            "network_id": o.network_id,
            "variant_id": o.variant_id,
            "state_id": o.state_id,
            "origin": o.origin,
            "seed": o.seed,
            "profile_time_s": o.profile_time_s,
            "profile_source": o.profile_source,
            "der_penetration": o.der_penetration,
            "source_v_pu": o.source_v_pu if o.load_p_w.size else None,
            "pf_status": o.pf_status,
            "pf_message": o.message,
            "n_off_setpoint": o.n_off_setpoint,
            "pf_iterations": o.pf_iterations,
            "residual_rel": o.residual_rel,
            "t_solve_s": o.t_solve_s,
            "t_extract_s": o.t_extract_s,
            "generator_version": versions["generator"],
            "schema_version": schema.SCHEMA_VERSION,
            "engine_versions": engines,
            "config_hash": config_hash,
        }
        for o in outcomes
    ]
    tables = {
        "scenarios/scenario_meta": pa.Table.from_pylist(
            meta, schema=schema.scenario_meta(precision)
        )
    }

    drawn = [o for o in outcomes if o.load_p_w.size]
    tables["scenarios/loads"] = concat_cast(
        [
            pa.table(
                {
                    "scenario_id": np.full(load_idx.size, o.scenario_id, dtype=np.int64),
                    "element_idx": load_idx,
                    "p_w": o.load_p_w,
                    "q_var": o.load_q_var,
                    "zip": pa.nulls(load_idx.size, pa.string()),
                },
            )
            for o in drawn
        ],
        schema.loads(precision),
    )
    der_parts = []
    for o in drawn:
        p, q = der_setpoints(o.pv_p_w, o.pv_q_var, o.bess_p_w, o.dg_p_w)
        if p is None:
            continue
        # storage: its p; DG: its available capacity
        p_avail = [a for a in (o.pv_p_avail_w, o.bess_p_w, o.dg_p_avail_w) if a is not None]
        der_parts.append(
            pa.table(
                {
                    "scenario_id": np.full(der_idx.size, o.scenario_id, dtype=np.int64),
                    "element_idx": der_idx,
                    "der_type": der_type,
                    "p_w": p,
                    "q_var": q,
                    "p_avail_w": np.concatenate(p_avail),
                    "s_rated_va": rated,
                },
            )
        )
    tables["scenarios/der"] = concat_cast(der_parts, schema.der(precision))
    converged = [o for o in outcomes if o.pf_status == "converged"]
    n_nodes = len(runner.model.nodes)
    node_idx = np.arange(n_nodes, dtype=np.int32)
    node_parts, element_parts = [], []
    plan = runner.plan
    for o in converged:
        assert o.v is not None
        node_parts.append(
            pa.table(
                {
                    "scenario_id": np.full(n_nodes, o.scenario_id, dtype=np.int64),
                    "node_idx": node_idx,
                    "v_re": o.v.real,
                    "v_im": o.v.imag,
                }
            )
        )
        if element_currents:
            assert o.i_pd is not None
            element_parts.append(
                pa.table(
                    {
                        "scenario_id": np.full(
                            plan.element_idx.size, o.scenario_id, dtype=np.int64
                        ),
                        "element_idx": plan.element_idx,
                        "terminal": plan.terminal,
                        "position": plan.position,
                        "i_re": o.i_pd.real,
                        "i_im": o.i_pd.imag,
                    }
                )
            )
    tables["scenarios/node_results"] = concat_cast(node_parts, schema.node_results(precision))
    if element_currents:
        target = schema.element_results(precision)
        tables["scenarios/element_results"] = concat_cast(element_parts, target)
    return tables


def concat_cast(parts: list[pa.Table], target: pa.Schema) -> pa.Table:
    """Concatenate and cast to ``target`` (precision, nullability); an empty table if no part."""
    if not parts:
        return target.empty_table()
    return pa.concat_tables([p.cast(target) for p in parts])
