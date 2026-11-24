# Third-party code

Code adapted from other projects, with its origin, licence and changes. Each adapted file says so
in its docstring.

| File | Origin | Licence | Changes |
|---|---|---|---|
| `gridfm_datakit_distr/config/value_spec.py` | [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) `gridfm_datakit/utils/value_spec.py`, commit 1b974c0 | Apache License 2.0 | typed for `mypy --strict`; path-less parser `parse_spec`; `to_config` |
| `tests/test_value_spec.py` | gridfm-datakit `tests/test_value_spec.py`, commit 1b974c0 | Apache License 2.0 | imports and paths adapted; tests of the additions |
| `gridfm_datakit_distr/io/datakit_columns.py` | gridfm-datakit `gridfm_datakit/utils/column_names.py` (column lists, verbatim), `gridfm_datakit/save.py` (`INT_COLUMNS`, the `scenario` and `scenario_partition` columns) and `gridfm_datakit/utils/utils.py` (`n_scenario_per_partition`), commit 1b974c0 | Apache License 2.0 | lists unchanged; constants named; `FILES` added |

The Apache License 2.0 text is in `LICENSE`. If the project's own licence changes, a copy of the
Apache License 2.0 stays with these files.

# Third-party data

Numbers derived from other parties' data. Only the derived numbers are in this repository.

| File | Origin | Licence | Derivation |
|---|---|---|---|
| `gridfm_datakit_distr/variants/upstream.py` (`RATING_COUNTS`) | Electricity North West Limited open data: the distribution-transformer capacity register (`sp-enw-capacity-distribution-tx`) and the smart-meter installation counts per substation (`smart-meter-installation`) | CC BY 4.0 | the two joined on the substation number; single-transformer substations with a customer count, ratings of 25 kVA and above; counted per rating and per octave of the customer count |
