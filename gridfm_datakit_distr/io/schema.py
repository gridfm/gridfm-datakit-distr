"""PyArrow schemas of the dataset tables, the single source of truth (see docs/manual/outputs.md).

Every table written by the package is checked against these schemas. A change to an existing
schema gets a new ``SCHEMA_VERSION`` and a migration note.
"""

from __future__ import annotations

from typing import Any, Literal

import pyarrow as pa

from gridfm_datakit_distr.io import datakit_columns as datakit

SCHEMA_VERSION = "0.7.0"


def _required(name: str, type_: pa.DataType) -> pa.Field[Any]:
    return pa.field(name, type_, nullable=False)


def _nullable(name: str, type_: pa.DataType) -> pa.Field[Any]:
    return pa.field(name, type_, nullable=True)


# ---------------------------------------------------------------- variant level (outputs.md §4)

NODES = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("node_idx", pa.int32()),
        _required("bus", pa.string()),
        _required("conductor", pa.string()),
        _required("opendss_node", pa.string()),
        _required("kv_base_ln", pa.float64()),
        _nullable("kv_nominal_ln", pa.float64()),
        _required("is_grounded", pa.bool_()),
        _nullable("r_ground_ohm", pa.float64()),
        _nullable("x_ground_ohm", pa.float64()),
        _required("is_source", pa.bool_()),
        _required("ground_by_convention", pa.bool_()),
        _nullable("backbone_bus", pa.string()),
    ]
)

ELEMENTS = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("element_idx", pa.int32()),
        _required("name", pa.string()),
        _required("element_class", pa.string()),
        _required("is_pd", pa.bool_()),
        _required("n_terminals", pa.int8()),
        _required("n_conductors", pa.int8()),
        _required("enabled", pa.bool_()),
        _required("is_tap_dependent", pa.bool_()),
        _nullable("rating_a", pa.float32()),
        _nullable("rating_va", pa.float32()),
    ]
)

ELEMENT_TERMINALS = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("element_idx", pa.int32()),
        _required("terminal", pa.int8()),
        _required("position", pa.int8()),
        _required("node_idx", pa.int32()),
    ]
)

YPRIM = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("element_idx", pa.int32()),
        _required("row", pa.int16()),
        _required("col", pa.int16()),
        _required("g", pa.float64()),
        _required("b", pa.float64()),
    ]
)

SOURCE = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("element_idx", pa.int32()),
        _required("position", pa.int8()),
        _required("node_idx", pa.int32()),
        _required("e_re", pa.float64()),
        _required("e_im", pa.float64()),
    ]
)
"""``variants/source``: the Vsource's ideal voltage (V, at 1 pu) per conductor of its terminal 1."""

SOURCE_YPRIM = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("row", pa.int16()),
        _required("col", pa.int16()),
        _required("g", pa.float64()),
        _required("b", pa.float64()),
    ]
)
"""``variants/source_yprim``: the Vsource's admittance (S) between the ideal voltages and its
nodes, non-zero entries; ``row``/``col`` are ``position`` values of ``variants/source``."""

VARIANTS = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("network_id", pa.string()),
        _required("variant_idx", pa.int32()),
        _required("seed", pa.uint64()),
        _required("perturbation", pa.string()),
        _required("perturbation_script_sha1", pa.string()),
        _required("n_nodes", pa.int32()),
        _required("n_elements", pa.int32()),
    ]
)
"""``variants/variants``: one row per accepted variant; ``perturbation`` is JSON."""

# ---------------------------------------------------- network-state level (outputs.md §5)

YPRIM_STATE = pa.schema([_required("state_id", pa.string()), *YPRIM])

STATES = pa.schema(
    [
        _required("state_id", pa.string()),
        _required("variant_id", pa.string()),
        _required("taps", pa.string()),
        _required("capacitor_states", pa.string()),
        _required("n_scenarios", pa.int32()),
    ]
)
"""``states/states``: one row per distinct network state of a variant; JSON columns."""

YBUS_STATE = pa.schema(
    [
        _required("state_id", pa.string()),
        _required("row", pa.int32()),
        _required("col", pa.int32()),
        _required("g", pa.float64()),
        _required("b", pa.float64()),
    ]
)
"""``states/ybus``: the entries of a state's network admittance matrix that differ from its
variant's (``variants/ybus``; a zero where an entry vanishes), node indices of the variant;
``views/kron/ybus``: the Kron view's whole matrix of a state."""

VARIANT_YBUS = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("row", pa.int32()),
        _required("col", pa.int32()),
        _required("g", pa.float64()),
        _required("b", pa.float64()),
    ]
)
"""``variants/ybus``: network admittance matrix of the variant as compiled (its initial network
state), node indices of the variant."""

# ---------------------------------------------------------------- network level (outputs.md §3)

CATALOG_NETWORKS = pa.schema(
    [
        _required("network_id", pa.string()),
        _required("network_idx", pa.int32()),
        _nullable("representation", pa.string()),
        _required("source", pa.string()),
        _required("original_format", pa.string()),
        _nullable("converter", pa.string()),
        _required("license", pa.string()),
        _required("n_buses", pa.int32()),
        _required("n_nodes", pa.int32()),
        _required("n_elements", pa.int32()),
        _required("neutral_representation", pa.string()),
        _required("has_split_phase", pa.bool_()),
        _required("s_base_va", pa.float64()),
        _required("frequency_hz", pa.float64()),
        _required("engine_version", pa.string()),
        _required("passport_path", pa.string()),
        _required("source_sha256", pa.string()),
    ]
)

# ---------------------------------------------------------------- scenario level (outputs.md §6)

Precision = Literal["mixed", "full"]


def _real(precision: Precision) -> pa.DataType:
    """float32 with ``mixed`` precision (currents, powers, setpoints), else float64."""
    return pa.float32() if precision == "mixed" else pa.float64()


def scenario_meta(precision: Precision) -> pa.Schema:
    real = _real(precision)
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("network_id", pa.string()),
            _required("variant_id", pa.string()),
            _nullable("state_id", pa.string()),
            _nullable("origin", pa.string()),
            _required("seed", pa.uint64()),
            _nullable("profile_time_s", pa.int64()),
            _nullable("pv_time_utc_s", pa.int64()),
            _required("profile_source", pa.string()),
            _required("der_penetration", real),
            _nullable("source_v_pu", real),
            _nullable("source_u2", real),
            _nullable("source_u2_angle_rad", real),
            _required("pf_status", pa.string()),
            _nullable("pf_message", pa.string()),
            _required("n_off_setpoint", pa.int32()),
            _nullable("n_pv_tripped", pa.int32()),
            _required("pf_iterations", pa.int16()),
            _nullable("residual_rel", real),
            _required("t_solve_s", real),
            _required("t_extract_s", real),
            _required("generator_version", pa.string()),
            _required("schema_version", pa.string()),
            _required("engine_versions", pa.string()),
            _required("config_hash", pa.string()),
        ]
    )


def node_results(precision: Precision) -> pa.Schema:
    """Node voltages; float64 in both precisions. Injections are ``Y_state V``."""
    del precision
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("node_idx", pa.int32()),
            _required("v_re", pa.float64()),
            _required("v_im", pa.float64()),
        ]
    )


def element_results(precision: Precision) -> pa.Schema:
    real = _real(precision)
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("element_idx", pa.int32()),
            _required("terminal", pa.int8()),
            _required("position", pa.int8()),
            _required("i_re", real),
            _required("i_im", real),
        ]
    )


def loads(precision: Precision) -> pa.Schema:
    real = _real(precision)
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("element_idx", pa.int32()),
            _required("p_w", real),
            _required("q_var", real),
            _nullable("zip", pa.string()),
        ]
    )


def der(precision: Precision) -> pa.Schema:
    real = _real(precision)
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("element_idx", pa.int32()),
            _required("der_type", pa.string()),
            _required("p_w", real),
            _required("q_var", real),
            _required("p_avail_w", real),
            _required("s_rated_va", real),
            _nullable("tripped", pa.bool_()),
        ]
    )


def opf_meta(precision: Precision) -> pa.Schema:
    """``opf/opf_meta``: one row per scenario of a dataset with OPF labels."""
    real = _real(precision)
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("opf_status", pa.string()),
            _nullable("opf_message", pa.string()),
            _required("opf_class", pa.string()),
            _required("cost_preset", pa.string()),
            _required("source_c1", pa.float64()),
            _required("source_c2", pa.float64()),
            _nullable("objective", pa.float64()),
            _nullable("source_p_w", pa.float64()),
            _nullable("iterations", pa.int32()),
            _nullable("t_solve_s", real),
            _nullable("curtailment_w", real),
            _nullable("verified", pa.bool_()),
            _nullable("verify_reason", pa.string()),
            _nullable("verify_max_dv_pu", pa.float64()),
            _nullable("verify_max_bound_excess_pu", pa.float64()),
            _nullable("controls_hold", pa.bool_()),
            _nullable("n_vmin_active", pa.int32()),
            _nullable("n_vmax_active", pa.int32()),
            _nullable("n_rating_active", pa.int32()),
            _nullable("opf_start", pa.string()),
        ]
    )


def opf_der(precision: Precision) -> pa.Schema:
    """``opf/opf_der``: the dispatch of each PV unit, for the solved scenarios."""
    real = _real(precision)
    return pa.schema(
        [
            _required("scenario_id", pa.int64()),
            _required("element_idx", pa.int32()),
            _required("p_w", real),
            _required("q_var", real),
            _required("q_max_var", real),
            _required("c1", pa.float64()),
            _required("c2", pa.float64()),
        ]
    )


SCENARIO_TABLES = {
    "scenarios/scenario_meta": scenario_meta,
    "scenarios/node_results": node_results,
    "scenarios/element_results": element_results,
    "scenarios/loads": loads,
    "scenarios/der": der,
    "opf/opf_meta": opf_meta,
    "opf/opf_der": opf_der,
    "opf/opf_node_results": node_results,
}
"""Scenario-level tables (one part per chunk) -> schema for a precision."""

# ---------------------------------------------------------------- Kron view (outputs.md §7.1)

KRON_NODES = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("node_idx_kron", pa.int32()),
        _required("bus", pa.string()),
        _required("conductor", pa.string()),
        _required("node_idx_full", pa.int32()),
    ]
)
KRON_NODE_RESULTS = pa.schema(
    [
        _required("scenario_id", pa.int64()),
        _required("node_idx_kron", pa.int32()),
        _nullable("v_re", pa.float64()),
        _nullable("v_im", pa.float64()),
        _required("view_status", pa.string()),
    ]
)
"""``views/kron/node_results``: rows for every scenario whose full model converged; the view's
voltages are null when the view itself did not converge at its setpoints (``view_status``)."""

# ----------------------------------------- positive-sequence view (outputs.md §7.2), datakit's


def _datakit(table: str, extras: list[pa.Field[Any]]) -> pa.Schema:
    """datakit's columns of ``table`` in its order and types (int64, int8, else float64), then
    the ``x_`` extras."""
    columns = {
        "bus": datakit.BUS_COLUMNS,
        "gen": datakit.GEN_COLUMNS,
        "branch": datakit.BRANCH_COLUMNS,
        "y_bus": datakit.YBUS_COLUMNS,
    }[table]
    ints = datakit.INT_COLUMNS[table]
    types = {"int64": pa.int64(), "int8": pa.int8()}
    fields = [_required(c, types.get(ints.get(c, ""), pa.float64())) for c in columns]
    return pa.schema(fields + extras)


_X_SCENARIO = _required("x_scenario_id", pa.int64())
DATAKIT_BUS = _datakit(
    "bus",
    [
        _X_SCENARIO,
        _required("x_truth_Vm", pa.float64()),
        _required("x_truth_Va", pa.float64()),
        _required("x_der_p_mw", pa.float64()),
        _required("x_der_q_mvar", pa.float64()),
        _required("x_der_p_avail_mw", pa.float64()),
    ],
)
DATAKIT_GEN = _datakit("gen", [_X_SCENARIO])
DATAKIT_BRANCH = _datakit("branch", [_X_SCENARIO, _required("x_vector_shift_deg", pa.float64())])
DATAKIT_YBUS = _datakit("y_bus", [_X_SCENARIO])
DATAKIT_TABLES = {
    "bus": DATAKIT_BUS,
    "gen": DATAKIT_GEN,
    "branch": DATAKIT_BRANCH,
    "y_bus": DATAKIT_YBUS,
}
"""datakit's tables of the positive-sequence view as ``export-posseq`` writes them (not dataset
tables)."""

# The positive-sequence view's own tables, written by ``views/posseq_export`` to its working
# directory and composed into datakit's tables: export intermediates, not dataset tables.


def _subset(table: pa.Schema, names: list[str], key: list[pa.Field[Any]]) -> pa.Schema:
    return pa.schema(key + [table.field(n) for n in names])


POSSEQ_BUS_RESULTS = pa.schema(
    [_required("scenario_id", pa.int64())]
    + [DATAKIT_BUS.field(n) for n in ("bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va")]
    + [
        DATAKIT_BUS.field(n)
        for n in DATAKIT_BUS.names
        if n.startswith("x_") and n != "x_scenario_id"
    ]
)
"""``views/posseq/bus_results``: what changes per scenario, for the converged views."""
_STATE = [_required("state_id", pa.string())]
POSSEQ_BUSES = _subset(
    DATAKIT_BUS,
    ["bus", "PQ", "PV", "REF", "vn_kv", "min_vm_pu", "max_vm_pu", "GS", "BS"],
    _STATE,
)
POSSEQ_BRANCHES = _subset(
    DATAKIT_BRANCH,
    [
        n
        for n in DATAKIT_BRANCH.names
        if n not in ("load_scenario_idx", "pf", "qf", "pt", "qt", "x_scenario_id")
    ],
    _STATE,
)
POSSEQ_YBUS = _subset(DATAKIT_YBUS, ["index1", "index2", "G", "B"], _STATE)
POSSEQ_GENERATORS = _subset(
    DATAKIT_GEN,
    [
        n
        for n in DATAKIT_GEN.names
        if n not in ("load_scenario_idx", "p_mw", "q_mvar", "x_scenario_id")
    ],
    [_required("variant_id", pa.string())],
)
POSSEQ_STATE_TABLES = {
    "views/posseq/buses": POSSEQ_BUSES,
    "views/posseq/branches": POSSEQ_BRANCHES,
    "views/posseq/ybus": POSSEQ_YBUS,
}
"""The view's network per state (merged at finalisation like the other state tables)."""
POSSEQ_SCENARIOS = pa.schema(
    [
        _required("scenario_id", pa.int64()),
        _required("variant_id", pa.string()),
        _required("view_status", pa.string()),
        _nullable("view_message", pa.string()),
    ]
)
"""``views/posseq/scenarios``: every scenario whose full model converged and its view's status."""
POSSEQ_MAPPING = pa.schema(
    [
        _required("variant_id", pa.string()),
        _required("posseq_bus", pa.int64()),
        _required("backbone_bus", pa.string()),
        _required("aggregated_buses", pa.string()),
    ]
)
POSSEQ_TABLES = {
    "views/posseq/bus_results": POSSEQ_BUS_RESULTS,
    **POSSEQ_STATE_TABLES,
    "views/posseq/generators": POSSEQ_GENERATORS,
    "views/posseq/scenarios": POSSEQ_SCENARIOS,
    "views/posseq/mapping": POSSEQ_MAPPING,
}

TABLES: dict[str, pa.Schema] = {
    "catalog/networks": CATALOG_NETWORKS,
    "views/kron/nodes": KRON_NODES,
    "views/kron/ybus": YBUS_STATE,
    "views/kron/node_results": KRON_NODE_RESULTS,
    "states/ybus": YBUS_STATE,
    "variants/variants": VARIANTS,
    "variants/nodes": NODES,
    "variants/elements": ELEMENTS,
    "variants/element_terminals": ELEMENT_TERMINALS,
    "variants/yprim": YPRIM,
    "variants/source": SOURCE,
    "variants/source_yprim": SOURCE_YPRIM,
    "variants/ybus": VARIANT_YBUS,
    "states/yprim_state": YPRIM_STATE,
    "states/states": STATES,
}
"""Table path (relative to the dataset root, without extension) -> schema."""


def check_schema(table: pa.Table, schema: pa.Schema, name: str) -> None:
    """Raise ``ValueError`` unless ``table`` has exactly ``schema`` (names, types, nullability)."""
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(
            f"table {name!r} does not match its schema:\n{table.schema}\nexpected:\n{schema}"
        )
