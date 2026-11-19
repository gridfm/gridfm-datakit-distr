# Configuration reference

<!-- Generated from `config/schema.py` by
     `uv run python -m gridfm_datakit_distr.config.reference > docs/manual/config.md`;
     do not edit. -->

A dataset is described by one YAML file (examples in `scripts/config/`), validated by
`gridfm_datakit_distr.config.load_config`. Unknown keys, non-finite numbers and values of the
wrong type are errors (`"1"` is not an integer; YAML reads `1e-10` as text, write `1.0e-10`);
every problem is reported as `<dotted.path>: <reason>`. Defaults leave the source networks
unchanged (no perturbation, no DER).

**Value specs** (gridfm-datakit's semantics): a number, or
`{distribution: normal, mean, std, min?, max?}` (truncated: out-of-range draws are redrawn, an
unbounded side ends 6 std beyond the mean), `{distribution: uniform, low, high}` (on
`[low, high)`), `{distribution: choice, values, weights?}`. A field's range applies to every
value a spec can draw: a normal needs an explicit `max` under a ceiling.

## Top level

A dataset generation config.

| Key | Type | Default | Description |
|---|---|---|---|
| `dataset_name` | string matching `^[A-Za-z0-9][A-Za-z0-9._-]*$` | required | Name of the dataset directory under `settings.data_dir`. |

## `networks`

Which catalog networks enter the dataset (`docs/manual/networks.md`).

| Key | Type | Default | Description |
|---|---|---|---|
| `networks.include` | list of strings (each of at least 1 character), at least 1 item | required | Catalog network ids or glob patterns (`fnmatch`, case-sensitive), e.g. `csiro_enwl/n01_*`. `<pattern>@<representation>` selects a named representation (`ieee8500@unbalanced`, `csiro_enwl/n01_f1@kron`) and overrides a plain match of the same network. Networks are ordered by id (seed tree). |
| `networks.neutral_representation` | one of `explicit`, `kron`, `as_is` | `explicit` | `kron` loads the Kron-reduced representation where the source provides one (CSIRO); `explicit` and `as_is` load each network's default model. |

## `networks.voltage_limits`

Voltage limits for statistics and OPF: a preset or explicit values, per unit of each
network's nominal line-to-neutral voltage (the base unless the catalog declares one).

| Key | Type | Default | Description |
|---|---|---|---|
| `networks.voltage_limits.preset` | one of `uk_esqcr`, `ansi_a` or null | null | `uk_esqcr`: 0.94 to 1.10 pu (UK ESQCR, 230 V +10 %/-6 %); `ansi_a`: 0.95 to 1.05 pu (ANSI C84.1 Range A). Sets `vmin_pu` and `vmax_pu`. |
| `networks.voltage_limits.vmin_pu` | number > 0 | `0.94` | Lower voltage limit (pu of the nominal voltage). |
| `networks.voltage_limits.vmax_pu` | number > 0 | `1.1` | Upper voltage limit (pu of the nominal voltage). |

## `normalisation`

Normalisation of the source models, recorded in the passports.

| Key | Type | Default | Description |
|---|---|---|---|
| `normalisation.s_base_va` | `auto` or a number > 0 (VA) | `auto` | Power base of the network (VA): per-unit base of the semantic model and the PMD export, scale of the positive-sequence view's source limits and balance tolerance (the view itself is per unit on 100 MVA). `auto`: the rating of the largest transformer of the source zone, else the total nominal load, else 1 MVA. |

## `normalisation.loads`

Load model applied before extraction.

| Key | Type | Default | Description |
|---|---|---|---|
| `normalisation.loads.model` | one of `constant_power`, `as_is` | `constant_power` | `constant_power`: every load becomes constant P, Q within [`vminpu`, `vmaxpu`]; `as_is`: the source's load models. |
| `normalisation.loads.vminpu` | number > 0 and < 1 | `0.6` | Below this voltage (pu) OpenDSS turns a load into a constant impedance. |
| `normalisation.loads.vmaxpu` | number > 1 | `1.5` | Above this voltage (pu) OpenDSS turns a load into a constant impedance. |

## `variants`

Perturbed copies of each network; each must converge and pass the Kirchhoff residual
check.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.n_variants` | integer ≥ 1 and ≤ 10000 | `4` | Variants per network (variant ids have four digits). |

## `variants.admittance`

Parameter perturbations of a variant; one draw per element.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.admittance.line_length_scale` | value spec, > 0 | `{distribution: uniform, low: 0.9, high: 1.1}` | Factor on each line's length. |
| `variants.admittance.neutral_impedance_scale` | value spec, > 0 | `1.0` | Factor on the neutral self and mutual impedances of explicit-neutral lines. |
| `variants.admittance.grounding_resistance_ohm` | value spec, > 0 or null | null | Resistance (Ω) of the existing grounding points; null keeps the source's. |

## `variants.phase_reallocation`

Moves single-phase customers to another phase available at their bus.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.phase_reallocation.enabled` | boolean | `true` | Reallocate phases in each variant. |
| `variants.phase_reallocation.fraction` | number ≥ 0 and ≤ 1 | `0.2` | Share of single-phase customers moved. |

## `der`

Distributed energy resources added to the networks.

## `der.pv`

Rooftop PV: placed per variant, output per scenario.

| Key | Type | Default | Description |
|---|---|---|---|
| `der.pv.penetration` | value spec, ≥ 0 and ≤ 1 | `{distribution: uniform, low: 0.0, high: 0.8}` | Share of customers with PV, drawn per variant. The default puts a voltage violation in about 6 % of the scenarios of a dataset of the whole catalog; 0 places no PV. |
| `der.pv.kwp` | value spec, > 0 | `{distribution: uniform, low: 2.0, high: 6.0}` | Peak power (kW) of each PV system. |
| `der.pv.power_factor` | value spec, > 0 and ≤ 1 | `1.0` | Power factor of each inverter, drawn per variant; below 1 the inverter absorbs reactive power, within its rating. |
| `der.pv.inverter_kva_per_kwp` | number > 0 | `1.1` | Inverter rating (kVA) per kWp of PV. |
| `der.pv.reactive_fraction` | value spec, ≥ 0 and ≤ 1 | `0.44` | Largest reactive power of each inverter in the OPF, as a fraction of its rating (IEEE 1547-2018 Category B: 0.44; 1: the rating's circle only); drawn per variant and unit. |
| `der.pv.reactive_threshold` | value spec, ≥ 0 and ≤ 1 | `0.05` | No reactive power in the OPF when a unit's available power is below this fraction of its rating (IEEE 1547-2018 Category B: 0.05; 0: always); drawn per variant and unit. |

## `scenarios`

Operating points of each variant.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.n_scenarios_per_variant` | integer ≥ 1 and ≤ 2147483648 | required | Scenarios per variant (scenario ids hold 31 bits of scenario index). |
| `scenarios.source_voltage_pu` | value spec, > 0 | `{distribution: uniform, low: 0.98, high: 1.05}` | Source voltage magnitude (pu), angle 0. |

## `scenarios.profiles`

Where load values come from.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.profiles.provider` | one of `synthetic`, `network_loadshapes`, `edp_smart_meter` | `synthetic` | `synthetic`; `network_loadshapes`: the loadshapes of the source model; `edp_smart_meter`: EDP's meter data. |
| `scenarios.profiles.same_timestamp` | boolean | `true` | All customers of a scenario share one timestamp (coincident demand). |

## `scenarios.profiles.synthetic`

Synthetic loads: a global factor per scenario and lognormal noise per customer.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.profiles.synthetic.global_scale` | value spec, ≥ 0 | `{distribution: uniform, low: 0.2, high: 1.2}` | Factor on every nominal load, drawn per scenario. |
| `scenarios.profiles.synthetic.local_sigma` | number ≥ 0 | `0.3` | Sigma of the lognormal noise on each customer's load. |
| `scenarios.profiles.synthetic.power_factor` | value spec, > 0 and ≤ 1 | `{distribution: uniform, low: 0.9, high: 0.99}` | Power factor of each load. |

## `scenarios.pv_profile`

PV output profile.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.pv_profile.provider` | one of `synthetic_clear_sky` | `synthetic_clear_sky` | Clear-sky profile reduced by cloudiness. |
| `scenarios.pv_profile.cloudiness` | value spec, ≥ 0 and ≤ 1 | `{distribution: uniform, low: 0.0, high: 1.0}` | Cloudiness, drawn per scenario. |

## `controls`

Regulator taps and capacitor states per scenario; labels always use
`ControlMode=OFF`.

| Key | Type | Default | Description |
|---|---|---|---|
| `controls.mode` | one of `static_settle`, `as_is` | `static_settle` | `static_settle`: settle with `ControlMode=STATIC`, move the regulated taps by the tap jitter and settle again from there, freeze the state; `as_is`: the taps and capacitor states as compiled. |
| `controls.tap_jitter_steps` | integer ≥ 0 | `2` | Largest move (tap steps) of each settled tap before the controls settle again from there; 0 settles once. |

## `labels`

What is solved for each scenario.

| Key | Type | Default | Description |
|---|---|---|---|
| `labels.pf` | boolean | `true` | Power-flow labels. |
| `labels.element_currents` | boolean | `false` | Also store the current of every power-delivery element at each terminal conductor (`scenarios/element_results`); otherwise they are recomputed exactly as Yprim · V. |

## `labels.opf`

OPF labels: the minimum-cost OPF of each scenario (docs/manual/opf.md).

| Key | Type | Default | Description |
|---|---|---|---|
| `labels.opf.enabled` | boolean | `false` | Solve and verify the OPF of every scenario (needs the `opf` extra and constant-power loads). |
| `labels.opf.costs` | one of `economic`, `sampled` | `economic` | Cost preset. `economic`: source 1 per W, PV units 0 (minimum import); `sampled`: coefficients drawn per scenario from `sampled`. |
| `labels.opf.time_limit_s` | number > 0 | `300.0` | Wall-clock limit of one OPF solve (s), Ipopt's `max_wall_time`. |

## `labels.opf.sampled`

Cost coefficients of the preset ``sampled``, drawn per scenario (``c2 p^2 + c1 p``, ``p``
in W): one pair for the source, one per PV unit.

| Key | Type | Default | Description |
|---|---|---|---|
| `labels.opf.sampled.source_c1` | value spec, ≥ 0 | `1.0` | Linear cost of the source's power (per W). |
| `labels.opf.sampled.source_c2` | value spec, ≥ 0 | `0.0` | Quadratic cost of the source's power (per W²). |
| `labels.opf.sampled.pv_c1` | value spec, ≥ 0 | `{distribution: uniform, low: 0.0, high: 1.0}` | Linear cost of each PV unit's power (per W), drawn per unit. |
| `labels.opf.sampled.pv_c2` | value spec, ≥ 0 | `0.0` | Quadratic cost of each PV unit's power (per W²). |

## `views`

Reduced views written next to the full model. The positive-sequence view is not
one: ``export-posseq`` computes it from a dataset.

| Key | Type | Default | Description |
|---|---|---|---|
| `views.kron` | boolean | `false` | Kron view (neutral eliminated), re-solved per scenario; ~1 kB per 100 phase nodes and scenario. |

## `settings`

Execution settings; the dataset does not depend on the process and context counts.

| Key | Type | Default | Description |
|---|---|---|---|
| `settings.data_dir` | string of at least 1 character | `./data` | Directory of the datasets (relative to the working directory). |
| `settings.seed` | integer ≥ 0 | required | Root of the seed tree. |
| `settings.num_processes` | integer ≥ 1 or null | null | Worker processes; null: one per core. Each runs one variant at a time, so at most networks x variants are busy. |
| `settings.contexts_per_process` | integer ≥ 1 | `1` | OpenDSS contexts (threads) per process. |
| `settings.chunk_size` | integer ≥ 1 | `500` | Scenarios per output chunk. |
| `settings.float_precision` | one of `mixed`, `full` | `mixed` | `mixed`: float32 for currents, powers and setpoints; `full`: float64 everywhere (docs/manual/outputs.md). |
| `settings.residual_sample_rate` | number ≥ 0 and ≤ 1 | `0.01` | Share of scenarios with the full KCL residual check. |
| `settings.overwrite` | boolean | `false` | Replace a dataset of the same name generated from another config. |
| `settings.validation_tolerance` | number > 0 and ≤ 1e-08 | `1e-10` | OpenDSS `Tolerance` of the conformance runs; at most 1e-8 for the setpoint check. |
