"""gridfm-datakit's output columns (outputs.md §7.2).

The AC column lists are copied verbatim from gridfm-datakit,
``gridfm_datakit/utils/column_names.py`` at commit 1b974c0 (identical at d1aaa48), Apache License
2.0, Copyright the gridfm-datakit authors (``THIRD_PARTY_NOTICES.md``). The integer columns and
the scenario and partition columns are those that ``gridfm_datakit/save.py`` adds on writing, at
the same commit; ``N_SCENARIO_PER_PARTITION`` is ``n_scenario_per_partition`` of
``gridfm_datakit/utils/utils.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------- column_names.py (verbatim lists)

GEN_COLUMNS = [
    "load_scenario_idx",
    "idx",
    "bus",
    "p_mw",
    "q_mvar",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
    "cp0_eur",
    "cp1_eur_per_mw",
    "cp2_eur_per_mw2",
    "in_service",
    "is_slack_gen",
]

BUS_COLUMNS = [
    "load_scenario_idx",
    "bus",
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "min_vm_pu",
    "max_vm_pu",
    "GS",
    "BS",
]

BRANCH_COLUMNS = [
    "load_scenario_idx",
    "idx",
    "from_bus",
    "to_bus",
    "pf",
    "qf",
    "pt",
    "qt",
    "r",
    "x",
    "b",
    "Yff_r",
    "Yff_i",
    "Yft_r",
    "Yft_i",
    "Ytf_r",
    "Ytf_i",
    "Ytt_r",
    "Ytt_i",
    "tap",
    "shift",
    "ang_min",
    "ang_max",
    "rate_a",
    "br_status",
]

YBUS_COLUMNS = [
    "load_scenario_idx",
    "index1",
    "index2",
    "G",
    "B",
]

# ---------------------------------------------------------------- save.py and utils.py

INT_COLUMNS = {
    "bus": {
        "load_scenario_idx": "int64",
        "bus": "int64",
        "PQ": "int8",
        "PV": "int8",
        "REF": "int8",
    },
    "gen": {
        "load_scenario_idx": "int64",
        "idx": "int64",
        "bus": "int64",
        "in_service": "int8",
        "is_slack_gen": "int8",
    },
    "branch": {
        "load_scenario_idx": "int64",
        "idx": "int64",
        "from_bus": "int64",
        "to_bus": "int64",
        "br_status": "int8",
    },
    "y_bus": {
        "load_scenario_idx": "int64",
        "index1": "int64",
        "index2": "int64",
    },
    "runtime": {
        "load_scenario_idx": "int64",
    },
}
"""Integer columns and their types; every other column is float64."""

SCENARIO_COLUMN = "scenario"
"""Inserted first on writing: the global scenario index, 0 .. n - 1 (int64)."""
PARTITION_COLUMN = "scenario_partition"
"""Hive partition key: ``scenario // N_SCENARIO_PER_PARTITION`` (int64)."""
N_SCENARIO_PER_PARTITION = 200
FILES = {
    "bus": "bus_data.parquet",
    "gen": "gen_data.parquet",
    "branch": "branch_data.parquet",
    "y_bus": "y_bus_data.parquet",
}
"""Partitioned parquet directory of each table under ``<data_dir>/<network>/raw``."""
