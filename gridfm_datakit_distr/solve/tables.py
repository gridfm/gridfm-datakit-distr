"""Tables of a variant and of a chunk of its scenarios (outputs.md §4-§6).

``variant_tables``: the variant's row and its model (nodes, elements, terminals, tap-independent
Yprims). ``state_tables``: the states a chunk met (row, tap-dependent Yprims, network admittance).
``chunk_tables``: ``scenario_meta`` for every scenario; ``loads`` and ``der`` (setpoints) for every
scenario with draws; ``node_results`` (and, when asked, ``element_results``) for converged
scenarios only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.writer import STATES_PART
from gridfm_datakit_distr.model import io_parquet
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.state import NetworkState
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner
from gridfm_datakit_distr.variants.generate import VariantResult, variants_table


def variant_tables(result: VariantResult) -> dict[str, pa.Table]:
    """``variants/variants`` and the variant's model tables."""
    if result.model is None:
        raise ValueError(f"{result.variant_id} failed: no tables")
    tables = {"variants/variants": variants_table([result])}
    tables.update(io_parquet.variant_level_tables(result.model))  # tap-dependent: per state
    return tables


def state_tables(states: dict[str, tuple[NetworkState, ElectricalModel]]) -> dict[str, pa.Table]:
    """Rows of the states met by a chunk (``states/states`` without the count)."""
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
        y = model.ybus().tocoo()
        ybus.append(
            pa.table(
                {
                    "state_id": [sid] * y.nnz,
                    "row": y.row.astype(np.int32),
                    "col": y.col.astype(np.int32),
                    "g": y.data.real,
                    "b": y.data.imag,
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
    element_currents: bool = True,
) -> dict[str, pa.Table]:
    """The scenario-level tables of a chunk; ``element_results`` only with
    ``element_currents`` (otherwise they are Yprim · V)."""
    index = {e.name: e.idx for e in runner.model.elements}
    load_idx = np.array([index[c.element] for c in runner.customers], dtype=np.int32)
    pv_idx = np.array(
        [index[u.generator] for u in (runner.pv.units if runner.pv is not None else ())],
        dtype=np.int32,
    )
    rated = np.array([u.s_rated_va for u in runner.pv.units] if runner.pv is not None else [])
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
    tables["scenarios/der"] = concat_cast(
        [
            pa.table(
                {
                    "scenario_id": np.full(pv_idx.size, o.scenario_id, dtype=np.int64),
                    "element_idx": pv_idx,
                    "der_type": ["pv"] * pv_idx.size,
                    "p_w": o.pv_p_w,
                    "q_var": o.pv_q_var,
                    "p_avail_w": o.pv_p_avail_w,
                    "s_rated_va": rated,
                },
            )
            for o in drawn
            if o.pv_p_w is not None
        ],
        schema.der(precision),
    )
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
