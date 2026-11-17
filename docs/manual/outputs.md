# Data schema — gridfm-datakit-distr

`schema_version: 0.3.0` · Status: in progress for v0.2. Any change gets a new `schema_version` and a migration note.

This document is the contract between the generator and its consumers (a future multiphase GENCO in gridfm-graphkit, baselines, benchmark users).

---

## 1. Conventions

**Four levels of data.** Static data is stored once at the level where it changes, never per scenario.

| Level | Key | Changes when | Examples |
|---|---|---|---|
| Network | `network_id` | never (source data) | catalog entry, passport |
| Variant | `variant_id` | topology, line/neutral/grounding parameters, phase allocation of customers change | nodes, elements, tap-independent Yprim |
| Network state | `state_id` | tap positions or capacitor states change | tap-dependent Yprim, network admittance matrix |
| Scenario | `scenario_id` | every sample | loads, DER outputs, PF/OPF solutions |

**Identifiers.**
- `network_id`: string from `networks/catalog.yaml`, e.g. `ieee13`, `csiro_enwl/n01_f1`, `edp/rural`.
- `variant_id`: `"{network_id}/v{k:04d}"`.
- `state_id`: first 16 hex chars of SHA-1 over (`variant_id`, tap vector rounded to 1e-9, capacitor states) — content-addressed, deduplicated.
- `scenario_id`: `int64`, unique within a dataset; assigned deterministically from the seed tree. Its network bits are the network's index in the dataset; the key of a scenario that is the same in every dataset holding its network is (`network_id`, `variant_idx`, `s`).
- `node_idx`, `element_idx`: `int32`, dense, 0-based, stable within a variant.
- Ground is never a node: terminal positions connected to ground have `node_idx = -1`.

**Units.** SI everywhere in raw tables: V, A, W, var, VA, Ω, S, m, Hz. Angles in radians. Only the positive-sequence export (§7.2) uses datakit's units: per unit with bases stated, MW, MVAr, degrees.

**Complex numbers.** Two columns: `<name>_re`, `<name>_im`.

**Precision.**
- `float64`: voltages, admittances (Yprim, Ybus), anything used in physics checks.
- `float32` (with `settings.float_precision: mixed`, the default): currents, powers, setpoints.
- `float_precision: full` → `float64` everywhere.

**Files.** Apache Parquet, zstd compression, row groups ≤ 128 MB, each written to a temporary name and renamed into place. Tables are written in parts, one per work unit: variant-level tables `part-{network_idx:05d}-{variant_idx:04d}.parquet`, scenario-level tables `part-{network_idx:05d}-{variant_idx:04d}-{chunk:05d}.parquet` (`network_idx`: the network's index in the dataset, `catalog/networks`). State-level tables are written in chunk parts during generation and merged into single files when the dataset is complete (a state met by several chunks is stored once).

**Conductor labels.** `a`, `b`, `c` for phases, `n` for neutral; North-American split-phase secondaries use `s1`, `s2`, `n`. OpenDSS conductor numbers are kept in `opendss_node`.

---

## 2. Directory layout

```
<data_dir>/<dataset_name>/
  manifest.json                    # config (resolved), config_hash, seeds, versions, chunk list + status
                                   # and peak memory of the process that ran each chunk (MiB)
  README.md                        # dataset card (generated)
  croissant.json                   # Croissant metadata (generated)
  catalog/
    networks.parquet
    passports/<network_id>.json
  variants/
    variants/part-*.parquet
    nodes/part-*.parquet
    elements/part-*.parquet
    element_terminals/part-*.parquet
    yprim/part-*.parquet           # tap-independent PD elements only
    source/part-*.parquet          # the voltage source: ideal voltages ...
    source_yprim/part-*.parquet    # ... behind its admittance
    semantic/<variant_id>.json     # semantic layer (PMD-ENGINEERING-aligned)
  states/
    states.parquet
    yprim_state.parquet            # tap/state-dependent elements
    ybus.parquet                   # network admittance matrix per state
  scenarios/
    scenario_meta/part-*.parquet
    node_results/part-*.parquet
    element_results/part-*.parquet  # only with labels.element_currents
    loads/part-*.parquet
    der/part-*.parquet
  views/
    kron/
      nodes/part-*.parquet           # per variant
      ybus.parquet                   # Kron-view admittance per state
      node_results/part-*.parquet
  opf/                             # with labels.opf.enabled (§8)
    opf_meta/part-*.parquet
    opf_der/part-*.parquet
    opf_node_results/part-*.parquet
  baselines/                       # not generated yet
    lindist_pf/part-*.parquet
    lindist_opf/part-*.parquet
  measurements/                    # not generated yet
    measurement_configs.parquet
  stats/
```

---

## 3. Network level

### `catalog/networks.parquet`

| Column | Type | Description |
|---|---|---|
| `network_id` | string | catalog key |
| `network_idx` | int32 | index of the network in the dataset (networks ordered by id; bits 47–62 of `scenario_id`) |
| `representation` | string, nullable | named representation loaded (e.g. `kron`), null for the default model |
| `source` | string | e.g. `ieee`, `csiro_enwl`, `edp`, `smart_ds` |
| `original_format` | string | `opendss` (others converted upstream) |
| `converter` | string, nullable | tool and version if converted (e.g. DiTTo) |
| `license` | string | SPDX id or free text |
| `n_buses`, `n_nodes`, `n_elements` | int32 | after normalisation |
| `neutral_representation` | string | `explicit`, `kron`, `mixed` |
| `has_split_phase` | bool | North-American secondaries present |
| `s_base_va` | float64 | per-feeder power base (VA): semantic model and PMD export; scale of the positive-sequence export's source limits and balance tolerance |
| `frequency_hz` | float64 | base frequency |
| `engine_version` | string | DSS C-API / AltDSS version string |
| `passport_path` | string | relative path |
| `source_sha256` | string | checksum of the fetched archive |

### `catalog/passports/<network_id>.json`

Free-form but must contain: `network_id`, `source_url`, `license`, `citation`, `checksums`, `original_format`, `converter`, `normalisation` (list of applied normalisations with parameters, e.g. `{"loads.vminpu": 0.6, "loads.vmaxpu": 1.5}`), `neutral_representation`, `slack_definition`, `known_differences` (vs other engines), `excluded_elements` (if any).

---

## 4. Variant level

### `variants/variants/part-*.parquet`

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `network_id` | string | |
| `variant_idx` | int32 | |
| `seed` | uint64 | 64-bit fingerprint of the variant's seed sequence (`generate_state(1, uint64)`); draws come from the sequence fixed by `settings.seed`, the network's id (two 32-bit words of its SHA-256) and the variant index, so a network's variants and scenarios do not depend on the other networks of the dataset |
| `perturbation` | string (JSON) | sampled parameters (line length scales, grounding resistances, phase reallocations, …) |
| `perturbation_script_sha1` | string | hash of the OpenDSS edit script that realises the variant |
| `n_nodes`, `n_elements` | int32 | |

### `variants/nodes/part-*.parquet`

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `node_idx` | int32 | |
| `bus` | string | OpenDSS bus name (lower case) |
| `conductor` | string | `a`, `b`, `c`, `n`, `s1`, `s2` |
| `opendss_node` | string | e.g. `671.2` |
| `kv_base_ln` | float64 | line-to-neutral base voltage of the bus (kV) |
| `kv_nominal_ln` | float64, nullable | nominal line-to-neutral voltage of the bus (kV) where the catalog declares one different from the base (the CSIRO feeders and the European LV feeder: 0.23 kV on a 0.416 kV base); null: the base. Voltage limits (`networks.voltage_limits`) are per unit of it |
| `is_grounded` | bool | conductor tied to ground at this bus |
| `r_ground_ohm`, `x_ground_ohm` | float64, nullable | grounding impedance if any |
| `is_source` | bool | node of the source (slack) bus |
| `backbone_bus` | string, nullable | three-phase backbone bus this node aggregates to in the positive-sequence export (§7.2); null without a three-phase backbone |

### `variants/elements/part-*.parquet`

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `element_idx` | int32 | |
| `name` | string | OpenDSS full name, e.g. `Line.650632` |
| `element_class` | string | `line`, `transformer`, `switch`, `capacitor`, `reactor`, `load`, `generator`, `pvsystem`, `storage`, `vsource` |
| `is_pd` | bool | power-delivery element (contributes to the network admittance) |
| `n_terminals`, `n_conductors` | int8 | |
| `enabled` | bool | |
| `is_tap_dependent` | bool | Yprim depends on the network state (transformers with tap control, switched capacitors) |
| `rating_a` | float32, nullable | normal ampacity per conductor (lines, switches), as the source declares it (on the line, its linecode or its wires); null where the source declares none (OpenDSS would assume 400 A) |
| `rating_va` | float32, nullable | apparent-power rating (transformers) |

### `variants/element_terminals/part-*.parquet`

One row per (element, terminal, position). `position` follows `CktElement.NodeOrder()`.

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `element_idx` | int32 | |
| `terminal` | int8 | 0-based terminal |
| `position` | int8 | 0-based conductor position within the terminal |
| `node_idx` | int32 | `-1` if grounded |

### `variants/yprim/part-*.parquet`

Non-zero entries of the primitive admittance matrix of each tap-independent PD element, in the element's local ordering (terminal-major, then position).

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `element_idx` | int32 | |
| `row`, `col` | int16 | local indices |
| `g`, `b` | float64 | S |

### `variants/source/part-*.parquet`

The voltage source (the Vsource) as ideal voltages behind an admittance: the source draws
`I = Y_s (source_v_pu · E - V)` into the network at its nodes, `V` the node voltages, `E` the
voltages below and `source_v_pu` the scenario's (`scenarios/scenario_meta`). One row per conductor
of its first terminal; its second terminal is ground (a network whose Vsource is not grounded
there, or with more than one, fails generation). With these and `states/ybus`, the setpoints of
`scenarios/loads` and `scenarios/der` determine the voltages of `scenarios/node_results`: the
reference loader `gridfm_datakit_distr.io.reader` and the power flow of
`gridfm_datakit_distr.validate.closed_loop` reproduce them without OpenDSS.

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `element_idx` | int32 | the Vsource element |
| `position` | int8 | 0-based conductor position within its first terminal |
| `node_idx` | int32 | |
| `e_re`, `e_im` | float64 | ideal source voltage at 1 pu (V): `BasekV` line-to-neutral, at `AngleDeg - 120°·position` |

### `variants/source_yprim/part-*.parquet`

Non-zero entries of `Y_s`, the block of the Vsource's Yprim at its first terminal.

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `row`, `col` | int16 | `position` values of `variants/source` |
| `g`, `b` | float64 | S |

### `variants/semantic/<variant_id>.json`

Semantic layer aligned with the PMD ENGINEERING data model (field names as in PMD: `bus.terminals`, `bus.grounded`, `bus.rg`, `bus.xg`, `line.f_bus`, `line.t_bus`, `line.f_connections`, `line.t_connections`, `line.rs`, `line.xs`, `line.g_fr`, `line.b_fr`, `line.g_to`, `line.b_to`, `line.length`, `line.cm_ub`, `transformer.bus`, `transformer.connections`, `transformer.configuration`, `transformer.vm_nom`, `transformer.sm_nom`, `transformer.xsc`, `transformer.rw`, `transformer.tm_set`, `transformer.tm_lb`, `transformer.tm_ub`, `load.bus`, `load.connections`, `load.configuration`, `load.model`, `load.pd_nom`, `load.qd_nom`, `load.vm_nom`, `solar/generator/storage` fields, `shunt.gs`, `shunt.bs`, `switch.state`, `voltage_source.vm`, `voltage_source.va`, `settings.sbase_default`, `settings.vbases_default`, `settings.voltage_scale_factor`, `settings.power_scale_factor`). Units as in PMD (scale factors stated in `settings`). This file is the input of the PMD exporter.

Normative definition: the pydantic models of `gridfm_datakit_distr.model.semantic`. Conventions:
terminals are OpenDSS node numbers; conductors tied to node 0 share one extra grounded terminal per bus
(4 unless taken; `rg = xg = 0`); delta windings and delta loads have no neutral connection; matrices are
lists of rows; series and shunt parameters come from the Yprims. Two fields are not PMD's:
`line.length_unit` (`"m"`, or `"none"` when neither the line nor its linecode declares a unit, as in
IEEE 37, and for series reactors and capacitors stored as lumped lines of length 1: only
`rs · length` etc. are then physical) and `transformer.leadlag` (`"lag"`/`"lead"`, the OpenDSS
delta-wye phase-shift convention, mapped to PMD's `polarity` and connection order by the exporter).
Single-phase transformers are not merged into banks.

---

## 5. Network-state level

### `states/states.parquet`

| Column | Type | Description |
|---|---|---|
| `state_id` | string | content hash |
| `variant_id` | string | |
| `taps` | string (JSON) | `{transformer_name: [tap per winding/phase]}` (per-unit ratio, rounded to 1e-9; the label solves of the state apply these taps) |
| `capacitor_states` | string (JSON) | `{capacitor_name: [0/1 per step]}` |
| `n_scenarios` | int32 | scenarios referencing this state |

### `states/yprim_state.parquet`

Same columns as `variants/yprim` plus `state_id`, for elements with `is_tap_dependent = true`.

### `states/ybus.parquet`

Network admittance matrix of the state, at node level (node indices of the variant). Includes all PD elements (lines, transformers at their taps, switches, reactors, capacitors at their state). Excludes power-conversion elements (loads, generators, PV, storage) and the source's internal impedance.

| Column | Type | Description |
|---|---|---|
| `state_id` | string | |
| `row`, `col` | int32 | node indices |
| `g`, `b` | float64 | S |

---

## 6. Scenario level

### `scenarios/scenario_meta`

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | `(network_idx << 47) \| (variant_idx << 31) \| s` |
| `network_id`, `variant_id` | string | |
| `state_id` | string, nullable | network state of the label solve; null if the controls did not settle or the scenario failed before |
| `origin` | string, nullable | how this scenario's controls reached its state: `static_settle` (settled on the scenario's setpoints), `static_settle+jitter` (the tap jitter moved the settled taps and the controls settled again, to another state), `as_is` (controls off, taps and capacitors as compiled); null when `state_id` is null. Scenarios sharing a state can have different origins |
| `seed` | uint64 | 64-bit fingerprint of the scenario's seed sequence (keyed by `settings.seed`, the network's id, the variant and scenario indices) |
| `profile_time_s` | int64, nullable | time of the load profile drawn, seconds since the start of the source's profile (no source is dated); null for synthetic profiles or one time per customer |
| `profile_source` | string | provider name |
| `der_penetration` | float32/64 | share of customers with PV in the variant |
| `source_v_pu` | float32/64, nullable | sampled source voltage magnitude |
| `pf_status` | string | `converged`, `off_setpoint` (converged, but a load or generator did not deliver its setpoint), `not_converged`, `error` |
| `pf_message` | string, nullable | exception type and message when `pf_status = error` |
| `n_off_setpoint` | int32 | loads and generators whose delivered power differs from the setpoint by more than 1e-6 relative |
| `pf_iterations` | int16 | |
| `residual_rel` | float32/64, nullable | power-balance residual of the stored solution, on the sampled scenarios only |
| `t_solve_s`, `t_extract_s` | float32/64 | |
| `generator_version`, `schema_version` | string | |
| `engine_versions` | string (JSON) | OpenDSS engine; CasADi (with its bundled Ipopt) with OPF labels |
| `config_hash` | string | |

Rows exist for every attempted scenario; only scenarios with `pf_status = converged` have result rows. `loads` and `der` rows (setpoints) exist for every scenario whose draws were made.

### `scenarios/node_results`

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `node_idx` | int32 | |
| `v_re`, `v_im` | float64 | node voltage to ground (V) |

Injections are not stored: the net current injected into the network by the
power-conversion elements and sources at each node is `I_inj = Y_state V` (`states/ybus`), and the
injected power is `V · conj(I_inj)`. At a node without power-conversion elements `I_inj` is zero up to
the power flow's residual (`scenario_meta.residual_rel`, on the sampled scenarios).

### `scenarios/element_results`

Currents of PD elements per terminal and conductor position, written only with `labels.element_currents: true`: they are `Yprim · V_local` exactly, from `variants/yprim`, `states/yprim_state` and the node voltages (the extraction's own formula; the tutorial shows it). Flows are derived (`S = V · conj(I)`), not stored.

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `element_idx` | int32 | |
| `terminal`, `position` | int8 | |
| `i_re`, `i_im` | float32/64 | current flowing into the element at that terminal conductor (A) |

### `scenarios/loads`

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `element_idx` | int32 | load element |
| `p_w`, `q_var` | float32/64 | setpoint at nominal voltage |
| `zip` | string (JSON), nullable | ZIP coefficients if not constant power |

Connection (bus, phases, wye/delta) is static and lives in the semantic layer. The power divides
equally between the branches of the element's terminal (`variants/element_terminals`): wye, when
the last conductor is a neutral or ground, one branch from each other conductor to it; delta on
three phases, between the pairs (a, b), (b, c), (c, a); one branch between two phases. The same
holds for `scenarios/der`. With the default `loads.model: constant_power` a converged scenario's
loads draw exactly their setpoints (`n_off_setpoint = 0`); with `as_is` the `zip` column is not
filled yet, and the stored tables do not determine the voltages.

### `scenarios/der`

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `element_idx` | int32 | DER element (OpenDSS `Generator` in v1) |
| `der_type` | string | `pv`, `bess`, `ev`, `hp` (EV and HP are modelled as loads in v1; kept for future use) |
| `p_w`, `q_var` | float32/64 | setpoint used in PF (generation positive) |
| `p_avail_w` | float32/64 | available active power (PV: from irradiance/profile × size) |
| `s_rated_va` | float32/64 | inverter rating |

---

## 7. Views

### 7.1 Kron view (`views/kron/`)

Network with the neutral eliminated by Kron reduction and neutral nodes merged into ground, **re-solved in OpenDSS** with the same injections.

- `nodes`: `variant_id`, `node_idx_kron`, `bus`, `conductor` (no `n`), `node_idx_full` (mapping to the full model).
- `ybus`: `state_id`, `row`, `col`, `g`, `b` (Kron-view admittance).
- `node_results`: `scenario_id`, `node_idx_kron`, `v_re`, `v_im` (view's own solution), `view_status`. The truth, the full model's phase-to-ground voltage of the same node, is `scenarios/node_results` at `node_idx_full` (not repeated).

The Kron view is off by default (`views.kron`): it costs about as much storage per phase node as the full model's voltages.

For networks whose `neutral_representation` is already `kron`, the view equals the full model and is stored only as a mapping.

### 7.2 Positive-sequence export (`export-posseq`; not stored in the dataset)

A single-phase equivalent of the three-phase backbone, **re-solved** in OpenDSS, in **gridfm-datakit's tables, column names, units and layout** (`utils/column_names.py` and `save.py` at datakit commit 1b974c0, columns identical at d1aaa48; vendored in `views/datakit_columns.py`), so that today's balanced GENCO can read it. It is not part of the dataset: `export-posseq` builds it from the stored truth after generation, one directory per network:

```
<out>/<network_id>/
  raw/                                    # graphkit's data_path
    bus_data.parquet/scenario_partition=<k>/part-0.parquet
    branch_data.parquet/…  gen_data.parquet/…  y_bus_data.parquet/…
    n_scenarios.txt
  posseq_scenarios.parquet                # the status of the view of every scenario
  posseq_mapping.parquet                  # the buses each bus of the view aggregates
```

- **Buses.** The backbone: buses reached from the source through series elements that cover phases a, b and c. `bus` numbers them breadth-first from the source: `bus` 0 is the source and the only `REF` bus, all others are `PQ`. Every other bus is aggregated into the backbone bus where its lateral starts (`posseq_mapping.parquet`).
- **Units (datakit's).** Powers in MW and MVAr (three-phase). `Vm` in pu of the bus's line-to-neutral base, `Va` in degrees in [-180, 180) relative to the source. `r`, `x`, `b`, `Y*`, `GS`, `BS`, `G`, `B` in pu on **100 MVA** (MATPOWER's and graphkit's default `baseMVA`) and the bus line-to-line base `vn_kv`. `rate_a` in MVA. `shift`, `ang_min`, `ang_max` in degrees.
- **Demand.** `Pd`, `Qd` are the bus's equivalent demand in the full model's solution: the power that the backbone elements deliver into the bus, which covers its loads, its laterals with their losses and shunts, and earthing. It is **net of DER**, since a datakit PQ bus has no generation; the DER output is in `x_der_*`.
- **Source.** Ideal, at |V1| of the full model's source bus. Its generator (`idx` 0, `is_slack_gen` 1) has limits ±`s_base_va` (`catalog/networks`, or the configured `normalisation.s_base_va`) and zero costs.
- **Branches.** `rate_a` is the sum of the elements' ratings (0 if none: unlimited); `br_status` 1; `ang_min`, `ang_max` ∓360. Transformers: `tap` is the no-load positive-sequence ratio at the from side, the vector-group shift is not modelled (`shift` = 0; the removed shift is in `x_vector_shift_deg`). Shunts at backbone buses (capacitors) are `GS`, `BS` at 1 pu. `min_vm_pu`, `max_vm_pu` are `networks.voltage_limits`.
- **Scenarios.** `scenario` = 0 … n − 1 over the scenarios whose view converged, in the order of `scenario_id`; partition k = `scenario` // 200. The stable key is `load_scenario_idx` = `x_scenario_id` = `scenario_id`: `scenario` changes when a dataset grows. Branch flows come from the voltages and the branch admittances (`S = V conj(Y V)` on 100 MVA).
- **Truth.** `x_truth_Vm`, `x_truth_Va` are the positive-sequence component of the full model's voltages at the bus, in the view's reference: the source at angle 0 and each zone rotated by its cumulative vector-group shift. The component is the same for phase-to-neutral and phase-to-ground voltages, because a common-mode voltage has no positive sequence; hence one pair of columns.
- **Extra columns** (prefix `x_`, ignored by graphkit). Bus: `x_scenario_id`, `x_truth_Vm`, `x_truth_Va`, and `x_der_p_mw`, `x_der_q_mvar`, `x_der_p_avail_mw` (DER output and available power aggregated at the bus). Branch: `x_scenario_id`, `x_vector_shift_deg`. Generator and admittance: `x_scenario_id`.

The export reproduces the view that the generation computed up to draft.7 to the last bit (IEEE 13, CSIRO n21_f3, EDP industrial). The exception is `x_der_*`, within 3e-10 relative, because the export reads the stored float32 setpoints.

`posseq_scenarios.parquet`, every scenario whose full model converged:

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `variant_id` | string | |
| `view_status` | string | `converged`, `off_setpoint` (a bus's power mismatch above 1e-6 `s_base_va`: a load left its voltage window), `not_converged`, `error` (e.g. a network without a three-phase backbone) |
| `view_message` | string, nullable | |

`posseq_mapping.parquet`:

| Column | Type | Description |
|---|---|---|
| `variant_id` | string | |
| `posseq_bus` | int64 | `bus` of the view |
| `backbone_bus` | string | OpenDSS bus name |
| `aggregated_buses` | string (JSON) | buses aggregated into it (its laterals) |

---

## 8. OPF

Written with `labels.opf.enabled` (the problem, its scope and its statuses: [OPF labels](opf.md)). Each scenario's OPF is solved at the scenario's network state, loads and source voltage; the PV units are dispatched. Parts and keys as the scenario tables (§1, §6).

### `opf/opf_meta`

One row per scenario of the dataset.

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `opf_status` | string | Ipopt's return status of the solve (`Solve_Succeeded` is the only label; e.g. `Infeasible_Problem_Detected`, `Maximum_WallTime_Exceeded`, `Maximum_Iterations_Exceeded`); `not_supported` (network outside the OPF's scope); `no_pf_label` (no power-flow label, not solved); `error` |
| `opf_message` | string, nullable | the reason with `not_supported`, the exception with `error` |
| `opf_class` | string | `A`: the uncontrolled power flow keeps every phase-to-neutral voltage within the OPF's bounds and every rated transformer within its rating; `B`: it does not; `C`: no power-flow label |
| `cost_preset` | string | `economic` (source 1 per W, PV units 0) or `sampled` (drawn per scenario, `labels.opf.sampled`) |
| `source_c1`, `source_c2` | float64 | the source's cost coefficients, `c2 p² + c1 p` with `p` the power the source delivers (W): per W, per W² |
| `objective` | float64, nullable | `Σ c2 p² + c1 p` over the source and the PV units at the returned point (W with `economic`); null without a solve |
| `source_p_w` | float64, nullable | the active power the source delivers at the returned point (W), the `p` of its cost: with `opf_der` it reproduces `objective`; null without a solve |
| `iterations` | int32, nullable | Ipopt iterations |
| `t_solve_s` | float32/64, nullable | wall time of the Ipopt solve (s) |
| `curtailment_w` | float32/64, nullable | `Σ (p_pf - p_opf)` over the PV units (W), against the power-flow dispatch (`scenarios/der.p_w`); round-off of the order of 1e-3 W either side |
| `verified` | bool, nullable | the dispatch passes the OpenDSS verification; null unless `Solve_Succeeded` |
| `verify_reason` | string, nullable | the failed checks: power flow not converged, set points, voltages, voltage bounds |
| `verify_max_dv_pu` | float64, nullable | largest `\|V_opendss - V_opf\|` per unit of the node's line-to-neutral base; verified only up to 1e-6 |
| `verify_max_bound_excess_pu` | float64, nullable | largest excess of a phase-to-neutral voltage over its bounds in the OpenDSS solution, per unit of the phase node's nominal voltage; at most 1e-6 when verified |
| `controls_hold` | bool, nullable | on a network with regulators or switched capacitors: whether a solve with the controls on (`ControlMode=STATIC`) at the solved dispatch keeps the scenario's taps and steps; null without controls or without a verification. Not part of `verified`: the label is the optimum at the frozen state |
| `n_vmin_active`, `n_vmax_active` | int32, nullable | phase-to-neutral voltages at their lower and upper bounds at the solution (within 1e-6 relative); null unless `Solve_Succeeded` |
| `n_rating_active` | int32, nullable | transformers at their rating at the solution (within 1e-6 relative); null unless `Solve_Succeeded` |
| `opf_start` | string, nullable | `flat`, or `power_flow` when the flat start did not end `Solve_Succeeded` and the row comes from a second solve started at the scenario's power-flow solution (`t_solve_s` then counts both, `iterations` the second); null without a solve |

### `opf/opf_der`

The dispatch of every PV unit, for the scenarios with `opf_status = Solve_Succeeded`.

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `element_idx` | int32 | the PV generator |
| `p_w`, `q_var` | float32/64 | active and reactive power injected (W, var) |
| `q_max_var` | float32/64 | the largest \|q\| the unit could dispatch (var): `der.pv.reactive_fraction` of its rating, 0 when its available power is below `der.pv.reactive_threshold` of its rating |
| `c1`, `c2` | float64 | the unit's cost coefficients (per W, per W²) |

### `opf/opf_node_results`

Node voltages of the OPF solution, for the scenarios with `opf_status = Solve_Succeeded`.

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int64 | |
| `node_idx` | int32 | |
| `v_re`, `v_im` | float64 | node voltage to ground (V) |

---

## 9. Baselines (not generated yet)

- `baselines/lindist_pf`: `scenario_id`, `node_idx_kron`, `vm_pu` per node (LinDist3Flow gives magnitudes), `t_solve_s`, `status`.
- `baselines/lindist_opf`: defined with its ticket.

The Kron and positive-sequence views (§7) are themselves the model-free baselines for PF.

---

## 10. Measurement configurations (not generated yet)

### `measurements/measurement_configs.parquet`

| Column | Type | Description |
|---|---|---|
| `config_id` | string | |
| `variant_id` | string | |
| `sensor_idx` | int32 | |
| `location_kind` | string | `node`, `element_terminal`, `customer` |
| `node_idx` / `element_idx` / `terminal` | int32 / int32 / int8, nullable | |
| `quantity` | string | `v_mag`, `v_phasor`, `p`, `q`, `i_mag`, `i_phasor` |
| `sigma` | float32 | standard deviation in SI units (or relative, see `sigma_kind`) |
| `sigma_kind` | string | `absolute`, `relative` |
| `is_pseudo` | bool | pseudo-measurement (from profiles) |
| `aggregation_s` | int32 | averaging window (e.g. 900 for 15-min smart-meter data) |

Noisy measurement values are generated deterministically on demand from (`scenario_id`, `config_id`, seed) by `measurements.noise.sample(...)`; they are materialised only for released benchmark splits.
