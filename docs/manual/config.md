# Configuration reference

<!-- Generated from `config/schema.py` by
     `uv run python -m gridfm_datakit_distr.config.reference > docs/manual/config.md`;
     do not edit. -->

A generation config is one YAML file that describes one dataset. This page lists every key with
its type, range, default and effect, generated from the models that validate it.

## Overview

The sections follow the order in which a dataset is built:

1. `networks`, `normalisation`: select catalog networks, normalise their models
   ([Networks](networks.md)).
2. `variants`, `der`: perturb each network into variants and place PV ([Variants](variants.md)).
3. `scenarios`: on each variant, draw the loads, PV output and source voltage of each scenario
   ([Scenarios](scenarios.md)).
4. `controls`: settle each scenario's regulators and capacitors into a network state.
5. `labels`, `views`: solve the power flow at that state; optional element currents, OPF
   ([OPF labels](opf.md)) and Kron view.
6. `settings`: run in chunks and write ([Outputs](outputs.md)).

Four keys are required; every other key has a default:

```yaml
dataset_name: my_dataset
networks: {include: [ieee13]}
scenarios: {n_scenarios_per_variant: 100}
settings: {seed: 1}
```

Sample configs in `scripts/config/`:

| Config | Shows |
|---|---|
| `default.yaml` | a commented starting point; every optional key it sets is at its default |
| `ieee13_small.yaml` | the smallest power-flow dataset: IEEE 13, 2 variants of 20 scenarios |
| `ieee123.yaml` | regulators and capacitors settled per scenario; ANSI voltage limits |
| `csiro_feeder.yaml` | a four-wire LV feeder: phase reallocation, PV |
| `csiro_enwl_pf.yaml` | most keys written out; Kron view on |
| `edp_rural.yaml` | loads from EDP's smart-meter profiles; EN 50160 voltage limits |
| `opf_small.yaml` | OPF labels on one LV feeder in seconds |
| `opf_slice.yaml` | OPF labels on five LV feeders with PV that pushes voltages over the limit |
| `sample.yaml` | the v0.2 sample dataset: IEEE 13 and EDP rural, power-flow and OPF labels |
| `health.yaml` | a small catalog health run: every network but `csiro_enwl/n13_f4` (fails C1) |

## Checking a config

`gridfm_datakit_distr generate <config> --dry-run` validates the config, loads its networks (they
must be fetched) and prints the counts of scenarios and work units with time and storage
estimates; it writes nothing. An invalid config stops it with exit code 2 and every problem listed
at once, one line each: `<dotted.path>: <reason>` (list items as `[i]`). In Python,
`gridfm_datakit_distr.config.load_config(path)` runs the same config checks without the network
cache.

- Unknown keys are errors; a missing key takes its default.
- Types are not coerced: `"1"` is not a number, `true` is not an integer (an integer is a valid
  number); `.inf` and `.nan` are refused.
- YAML 1.1: an exponent needs a decimal point (`1.0e-10`; `1e-10` is read as text); a
  duplicated key is an error.
- Units are SI unless the key's name says otherwise (`_pu`, `kwp` in kW).

**Dataset identity.** The dataset's `manifest.json` stores the resolved config, every default
filled in; its `config_hash` covers every key except `settings.data_dir`, `num_processes`,
`contexts_per_process` and `overwrite`. When `generate` resumes, refuses or overwrites a dataset:
[Command-line reference](cli.md#generate).

## Value specs

A key of type *value spec* takes a number or a distribution, drawn anew each time its description
says (per variant, per scenario, per unit):

- `0.5`: always 0.5.
- `{distribution: uniform, low: 0.9, high: 1.1}`: uniform on [low, high); low < high.
- `{distribution: normal, mean: 1.0, std: 0.05, min: 0.9, max: 1.1}`: normal truncated to
  [min, max], each bound optional; std > 0; out-of-range draws are redrawn, never clipped.
- `{distribution: choice, values: [1, 5, 20], weights: [2, 1, 1]}`: one of `values`, with
  probabilities proportional to `weights` (non-negative, positive sum; default: equal).

The range in the Type column holds for every value a spec can draw. Under a lower bound, a fixed
number, a uniform's `low` or any choice below it is an error, while a normal is truncated at it.
Under an upper bound, the largest possible draw must respect it: a uniform's `high`, the largest
choice, the end of a normal. A normal's unbounded side ends 6 std beyond its mean (or beyond its
other bound, when that is further out), so under an upper bound it needs a `max` unless
mean + 6 std fits; a `[min, max]` outside mean ± 6 std is an error.

## Top level

Top-level keys. `networks`, `scenarios` and `settings` hold required keys; the other
sections may be omitted.

| Key | Type | Default | Description |
|---|---|---|---|
| `dataset_name` | string matching `^[A-Za-z0-9][A-Za-z0-9._-]*$` | required | Name of the dataset: its directory is `<settings.data_dir>/<dataset_name>`. |

## `networks`

Which catalog networks enter the dataset and which model of each is loaded
([Networks](networks.md)).

| Key | Type | Default | Description |
|---|---|---|---|
| `networks.include` | list of strings (each of at least 1 character), at least 1 item | required | Catalog network ids or glob patterns (`fnmatch`, case-sensitive; `*` also matches `/`), e.g. `ieee13`, `csiro_enwl/n01_*`, `*`. Each entry must match a network; overlapping matches count once. `<pattern>@<representation>` loads that alternative model of every network the pattern matches (an error where one lacks it or is already selected with another representation) and wins over a plain match of the same network: `four_wire` (the default), `kron`, `phase_to_neutral`, `modified_phase_to_neutral` on the CSIRO feeders, `unbalanced` on `ieee8500`; the alternatives are not yet validated. Patterns do not skip networks that fail the conformance suite: `*` selects `csiro_enwl/n13_f4`, whose variants do not converge. The selected networks are sorted by id: that order numbers them in scenario ids and file names (`network_idx`); a network's random draws depend on its id only. |
| `networks.neutral_representation` | one of `explicit`, `kron`, `as_is` | `explicit` | `kron`: the `kron` representation of every network that has one (the CSIRO feeders), the default model elsewhere. `explicit` and `as_is` both load each network's default model; `explicit` also has `generate` print a note naming the networks whose default model has no explicit neutral (the Kron-reduced IEEE feeders). An `@representation` in `include` overrides this key. |

### `networks.voltage_limits`

Voltage limits, per unit of each bus's nominal line-to-neutral voltage
(`variants/nodes.kv_nominal_ln`: the base unless the catalog declares a nominal, e.g. 230 V on
the 416 V CSIRO feeders). They do not change the power-flow labels: they bound the OPF, count
the violations in `stats` and set `min_vm_pu`, `max_vm_pu` of `export-posseq`. The defaults
are the `uk_esqcr` values.

| Key | Type | Default | Description |
|---|---|---|---|
| `networks.voltage_limits.preset` | one of `uk_esqcr`, `ansi_a` or null | null | `uk_esqcr`: 0.94 to 1.10 pu (UK ESQCR, 230 V +10 %/-6 %); `ansi_a`: 0.95 to 1.05 pu (ANSI C84.1 Range A). Sets `vmin_pu` and `vmax_pu`: giving either with another value is an error. null: `vmin_pu` and `vmax_pu` as given. |
| `networks.voltage_limits.vmin_pu` | number > 0 | `0.94` | Lower limit (pu of the nominal voltage); must be below `vmax_pu`. |
| `networks.voltage_limits.vmax_pu` | number > 0 | `1.1` | Upper limit (pu of the nominal voltage). |

## `normalisation`

Normalisation applied to every network and variant after compilation, recorded in the
passports (`catalog/passports`).

| Key | Type | Default | Description |
|---|---|---|---|
| `normalisation.s_base_va` | `auto` or a number > 0 (VA) | `auto` | Power base (VA) of each network: the per-unit base of the semantic model (`variants/semantic`, PMD-aligned), `catalog/networks.s_base_va` and the passports, and in `export-posseq` the source generator's limits and the balance tolerance of the positive-sequence view (itself per unit on 100 MVA). `auto`: per network, the rating of the largest transformer with a winding in the source zone, else the total nominal apparent power of the loads, else 1 MVA. A number: that base for every network. |

### `normalisation.loads`

Model of every enabled load. Outside its voltage window [`vminpu`, `vmaxpu`] OpenDSS turns
a load into a constant impedance and still reports convergence; every solve is checked against
the setpoints, so a scenario where a load or generator misses its setpoint (1e-6 relative) is
`off_setpoint`, without results. The PV units get the same window.

| Key | Type | Default | Description |
|---|---|---|---|
| `normalisation.loads.model` | one of `constant_power` | `constant_power` | `constant_power`, the only model: every load becomes constant P, Q (OpenDSS `model=1`), whatever its model in the source. |
| `normalisation.loads.vminpu` | number > 0 and < 1 | `0.6` | Lower end of the window (pu of each load's rated voltage). OpenDSS's own 0.95 would turn loads into impedances in ordinary undervoltage. |
| `normalisation.loads.vmaxpu` | number > 1 | `1.5` | Upper end of the window (pu of each load's rated voltage). |

## `variants`

Perturbed copies of each network ([Variants](variants.md)). Each variant draws from its own
streams; its base case must converge and pass the Kirchhoff residual check, else it is redrawn,
3 attempts at most, and then recorded as failed: its units fail and `generate` exits with an
error. The defaults perturb every variant; `line_length_scale: 1.0`,
`phase_reallocation.enabled: false` and `der.pv.penetration: 0` keep the source network.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.n_variants` | integer ≥ 1 and ≤ 10000 | `4` | Variants per network (ids `v0000` to `v9999`). |

### `variants.admittance`

Parameter perturbations, drawn per element and variant. A fixed `1.0` (null for the
resistance) keeps the source's values and draws nothing.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.admittance.line_length_scale` | value spec, > 0 | `{distribution: uniform, low: 0.9, high: 1.1}` | Factor on each line's length (switches excluded): its series impedance and shunt admittance scale with it. |
| `variants.admittance.neutral_impedance_scale` | value spec, > 0 | `1.0` | Factor β on the neutral of every line with a neutral conductor: its self impedance times β, the phase-neutral mutual impedances times √β. No effect on networks without an explicit neutral. |
| `variants.admittance.grounding_resistance_ohm` | value spec, > 0 or null | null | Resistance (Ω) of every earth electrode: `Rneut` of grounded wye windings and grounding reactors, except the neutral's bond at a source. null: the source's values. |

### `variants.phase_reallocation`

Moves single-phase loads (one phase conductor, at a bus with another phase) to another
phase of their bus, with their neutral; phase-to-phase loads stay. Drawn per variant.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.phase_reallocation.enabled` | boolean | `true` | Reallocate in each variant; `false` ignores `fraction`. |
| `variants.phase_reallocation.fraction` | number ≥ 0 and ≤ 1 | `0.2` | Share of the eligible loads moved (rounded to the nearest count), each to one of the other phases of its bus, chosen uniformly. |

### `variants.upstream`

The MV/LV substation behind the source of the CSIRO feeders
([Variants](variants.md#upstream-substation)), whose source is an ideal LV source: per
variant, a transformer rating drawn from those of ENWL substations with as many customers;
the source gets the transformer's and the MV network's impedances, and the other feeders of
the substation become loads at the source bus, scaled per scenario with the customers.

| Key | Type | Default | Description |
|---|---|---|---|
| `variants.upstream.enabled` | boolean | `false` | Give every variant its substation. Only for `csiro_enwl` networks (`generate` refuses others) and the `synthetic` load provider. `false` (default): the source as loaded, and every output is the same as without this section. |

## `der`

Distributed energy resources added to every variant.

### `der.pv`

Rooftop PV units, placed and sized per variant (`variants/variants.perturbation`), output
per scenario (`scenarios.pv_profile`). Every enabled load is a customer that can get one: the
unit sits on the load's nodes (after any phase move), constant P, Q within the load window.

| Key | Type | Default | Description |
|---|---|---|---|
| `der.pv.penetration` | value spec, ≥ 0 and ≤ 1 | `{distribution: uniform, low: 0.0, high: 0.8}` | Share of customers with PV, drawn per variant: each customer gets a unit with that probability, so the realised share (`scenario_meta.der_penetration`) varies around the draw. `0`: no PV. The default put a voltage violation in about 6 % of the scenarios of a dataset of the whole catalog when PV units were 2-6 kWp; not yet measured at the current sizes. |
| `der.pv.kwp` | value spec, > 0 | `{distribution: uniform, low: 0.5, high: 10.0}` | Peak power (kW) of each unit, drawn per customer and variant. The default spans UK installed sizes at both ends (sources in the Variants manual): its 10th percentile (1.45 kW) lies below the measured one (about 1.5 kW) and its 90th (9.05 kW) above it (6.3 to 8.9 kW). No export limit: a unit injects up to its rating. |
| `der.pv.power_factor` | value spec, > 0 and ≤ 1 | `1.0` | Power factor of each inverter in the power flow, drawn per unit and variant. Below 1 the inverter absorbs reactive power, `q = -p · tan(acos(pf))`, limited to its rating. |
| `der.pv.reactive_mode` | one of `power_factor`, `uniform` | `power_factor` | Reactive power of each inverter in the power flow. `power_factor` (default): from `power_factor`. `uniform`: drawn per unit and scenario in both directions, `q = u · L` with `u` uniform on [-1, 1) and `L` the smaller of the OPF's limit (`reactive_fraction` of the rating, none below `reactive_threshold`) and the rating's circle at the unit's active power: IEEE 1547-2018's constant reactive power mode with its setpoint drawn per scenario. It does not follow the voltage. Needs `power_factor: 1`. |
| `der.pv.inverter_kva_per_kwp` | number > 0 | `1.1` | Inverter rating (kVA) per kWp. Active power is clipped at the rating; with `synthetic_clear_sky` the available power never exceeds kWp, so from 1 up it never clips (a PVGIS hour can exceed kWp). |
| `der.pv.reactive_fraction` | value spec, ≥ 0 and ≤ 1 | `0.44` | Largest reactive power of each inverter in the OPF, and in the power flow with `reactive_mode: uniform`, as a fraction of its rating (IEEE 1547-2018 Category B: 0.44; 1: the rating's circle only); drawn per unit and variant. |
| `der.pv.reactive_threshold` | value spec, ≥ 0 and ≤ 1 | `0.05` | A unit whose available power is below this fraction of its rating gives no reactive power in the OPF, and in the power flow with `reactive_mode: uniform` (IEEE 1547-2018 Category B: 0.05; 0: no threshold); drawn per unit and variant. |

### `der.storage`

Battery storage units, placed and sized per variant, set-point per scenario
(`scenarios.storage_profile`). A unit is an OpenDSS generator `Generator.bess_<load>` on the
nodes of a customer with PV, with a signed constant power (positive: discharging into the
network). It is fixed at its set-point in the power flow and in the OPF: a snapshot has no
stored energy to value, so storage is an OPF control only over time sequences.

| Key | Type | Default | Description |
|---|---|---|---|
| `der.storage.share` | value spec, ≥ 0 and ≤ 1 | `0.0` | Share of the PV customers with a storage unit, drawn per variant: each PV customer gets one with that probability. `0` (default): no storage, and every output is the same as without this section. |
| `der.storage.kw` | value spec, > 0 or null | null | Power rating (kW) of each unit; its `s_rated_va`. null (default): from the peak power of its customer's PV unit, by the rule for customer batteries of NREL's SMART-DS, a synthetic US dataset: below 4 kWp 4 kW, from 4 to 10 kWp 8 kW, from 10 to 150 kWp 25 kW, from 150 kWp 100 kW (each lower bound included: our reading of the source's bands). A value spec: drawn per unit and variant instead. |

### `der.dg`

Dispatchable generators ([Variants](variants.md#dispatchable-generators)), as in the DG
case study of Geth, Claeys and Deconinck (PSCC 2022): one unit at every 4th load (in name
order), every unit with the same capacity `P^max`, the smallest at which they congest the
network (computed per network, recorded in `manifest.json`). A unit is an OpenDSS generator
`Generator.dg_<load>` on the load's nodes; its set-point and availability `a` are drawn per
scenario (`scenarios.dg_profile`) and the OPF dispatches it within `0 <= p, q <= a · P^max`.

| Key | Type | Default | Description |
|---|---|---|---|
| `der.dg.enabled` | boolean | `false` | Place the network's dispatchable generators in every variant. `false` (default): none, and every output is the same as without this section. |

## `scenarios`

Operating points of each variant ([Scenarios](scenarios.md)); each scenario draws from its
own streams.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.n_scenarios_per_variant` | integer ≥ 1 and ≤ 2147483648 | required | Scenarios per variant of each network. |
| `scenarios.source_voltage_pu` | value spec, > 0 | `{distribution: uniform, low: 0.98, high: 1.05}` | Positive-sequence magnitude of the source's ideal voltage (pu of the source's base voltage), drawn per scenario; angle 0. Balanced unless `source_unbalance` adds a negative sequence. |
| `scenarios.source_unbalance` | value spec, ≥ 0 and ≤ 1 | `0.0` | Negative-sequence ratio `u2` of the source's ideal voltage, the magnitude of its negative sequence `E2` over that of its positive sequence `E1` (behind the source impedance, not at the source bus), drawn per scenario; the angle of `E2` relative to `E1` is uniform in [0, 2π). 0: a balanced source. For coverage of background unbalance, `{distribution: uniform, low: 0, high: 0.02}`: EN 50160 bounds the 10-min negative-sequence ratio at 2 % for 95 % of a week and the CIGRE/CIRED C4.103 survey finds about 95 % of MV and LV sites below 2 % at the weekly 95th percentile (both from secondary sources: the standard's text was not available); uniform is a coverage choice, not a measured distribution. No zero sequence: IEC TR 61000-3-13 defines no compatibility level for it. Needs a three-phase source. |

### `scenarios.profiles`

Where each scenario's loads come from ([Scenarios](scenarios.md)).

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.profiles.provider` | one of `synthetic`, `network_loadshapes`, `edp_smart_meter` | `synthetic` | `synthetic`: nominal loads times random factors (section `synthetic`). `network_loadshapes`: the loads' daily (else yearly) loadshapes in the source model, at a profile time drawn uniformly over the shortest multi-point loadshape; loads without one keep their nominal (or one-point) value. `edp_smart_meter`: the same mechanism for the EDP networks, whose loadshapes are the smart-meter files; only the recorded `profile_source` differs. `generate` refuses either on a network without loadshapes before any work, also with `--dry-run`. |
| `scenarios.profiles.same_timestamp` | boolean | `true` | Loadshape providers only (ignored by `synthetic`). `true`: one profile time per scenario for all customers (coincident demand), whose time of day is the PV hour (the profile start taken as midnight). `false`: one time per customer, and a PV hour drawn on its own. |

#### `scenarios.profiles.synthetic`

Synthetic loads, `provider: synthetic` only (this section with another provider is an
error): `P = L · g · m · P_nominal` and lagging `Q = P · tan(acos(pf))`, with `L` per variant,
`g` per scenario, `m` and `pf` per customer and scenario. The loads' nominal reactive powers
are not used.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.profiles.synthetic.variant_scale` | value spec, > 0 | `{distribution: log_uniform, low: 0.3787878787878788, high: 2.64}` | Load level `L` of a network, drawn per variant: a factor on every nominal load of the variant, the other feeders' loads of `variants.upstream` included; recorded as `variant_scale` in `variants/variants.perturbation`. The networks with catalog `load_level: false` keep `L = 1`: measured nominal loads (EDP) and MV feeders (IEEE 13, 34, 37, 123, 8500) ([Networks](networks.md#load-level)). Default log-uniform on [1/2.64, 2.64], median 1: the spread of the peak load per customer across measured feeders ([Scenarios](scenarios.md#loads)). `1`: no load level (every network at its nominal level). |
| `scenarios.profiles.synthetic.global_scale` | value spec, ≥ 0 | `{distribution: uniform, low: 0.1, high: 1.2}` | Factor `g` on every nominal load, drawn per scenario. |
| `scenarios.profiles.synthetic.local_sigma` | number ≥ 0 | `0.3` | Sigma of the per-customer factor `m = exp(sigma z - sigma²/2)`, `z` standard normal, of mean 1 (so the mean load is `L · E[g] · P_nominal`); 0: no per-customer noise. |
| `scenarios.profiles.synthetic.power_factor` | value spec, > 0 and ≤ 1 | `{distribution: uniform, low: 0.9, high: 0.99}` | Power factor `pf` of each load, drawn per customer and scenario. |

### `scenarios.pv_profile`

PV output per scenario: available power = kWp · output per kWp of the provider ·
availability, injected up to the inverter rating ([Scenarios](scenarios.md)).

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.pv_profile.provider` | one of `synthetic_clear_sky`, `pvgis` | `synthetic_clear_sky` | `synthetic_clear_sky`: `max(0, cos(π (h - 12) / 12))` between 6:00 and 18:00 times the cloud factor, no date or location. The hour `h` is the load profile's time of day when the scenario has one, else uniform over 24 h: with `synthetic` loads half the scenarios fall at night, with no PV output. `pvgis`: one hour of the PVGIS series of the network's location (section `pvgis`), at the load profile's local time of day on a date drawn uniformly when the scenario has one, else any hour of the series. |
| `scenarios.pv_profile.cloudiness` | value spec, ≥ 0 and ≤ 1 | `{distribution: uniform, low: 0.0, high: 1.0}` | `synthetic_clear_sky` only (another value with `pvgis` is an error): cloud cover fraction `c`, drawn per scenario for the whole network; the clear-sky output is scaled by `1 - 0.75 c^3.4` (Kasten-Czeplak): 1 leaves 25 %. |
| `scenarios.pv_profile.availability` | value spec, ≥ 0 and ≤ 1 | `1.0` | Factor on each unit's available power, drawn per unit and scenario (shading, or 0 for a unit out of service). With 1 every unit of a variant delivers the same share of its kWp in a scenario; a distribution makes their outputs differ, e.g. `{distribution: choice, values: [0, 1], weights: [0.05, 0.95]}` for units out of service. |

#### `scenarios.pv_profile.pvgis`

Hourly PV output from PVGIS 5.3 of the EU Joint Research Centre, `provider: pvgis` only
(this section with another provider is an error). One `seriescalc` series per location (with
`orientation`, one per orientation class): the power `P` of a 1 kWp system, with PVGIS's
temperature model and system loss, for every hour of the years; `fetch --config` downloads it
into the cache ([Scenarios](scenarios.md#pv-output)).
The keys below are PVGIS's API parameters.

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.pv_profile.pvgis.locations` | mapping of network id or glob pattern to `{lat, lon, timezone}` | `{}` | Where each network is: a catalog network id or a glob pattern (`fnmatch`, as `networks.include`) mapped to `lat` (degrees north), `lon` (degrees east) and `timezone` (IANA name, for the local time of day). An id wins over patterns; two patterns matching one network are an error, as is a key matching no catalog network. No network has a built-in location (no source gives a point): `generate` refuses a selected network without one. |
| `scenarios.pv_profile.pvgis.startyear` | integer ≥ 2005 and ≤ 2023 | `2005` | First year of the series. |
| `scenarios.pv_profile.pvgis.endyear` | integer ≥ 2005 and ≤ 2023 | `2023` | Last year of the series (≥ `startyear`). |
| `scenarios.pv_profile.pvgis.loss` | number ≥ 0 and < 100 | `14.0` | System loss (%), PVGIS's recommended 14 by default. |
| `scenarios.pv_profile.pvgis.pvtechchoice` | one of `crystSi`, `crystSi2025`, `CIS`, `CdTe`, `Unknown` | `crystSi` | PV technology (PVGIS's default `crystSi`). |
| `scenarios.pv_profile.pvgis.mountingplace` | one of `free`, `building` | `free` | `free`: rack-mounted with air behind the modules (PVGIS's default); `building`: building-integrated, no air behind. PVGIS places roof-mounted systems somewhere between the two. |
| `scenarios.pv_profile.pvgis.optimalangles` | boolean | `true` | Use PVGIS's fixed slope and azimuth with the highest yearly energy at each location (the response reports them, `manifest.json`); `false` needs `angle` and `aspect`, or `orientation`. |
| `scenarios.pv_profile.pvgis.angle` | integer ≥ 0 and ≤ 90 or null | null | Slope of the modules from the horizontal (whole degrees: PVGIS rounds others); with `optimalangles: false` only. |
| `scenarios.pv_profile.pvgis.aspect` | integer ≥ -180 and ≤ 180 or null | null | Azimuth of the modules (whole degrees, 0 south, 90 west, -90 east); with `optimalangles: false` only. |

##### `scenarios.pv_profile.pvgis.orientation`

Orientation of each PV unit (`provider: pvgis` with `optimalangles: false` only): per
variant, each unit draws one (tilt, azimuth) pair uniformly among the pairs of the two lists
and reads the PVGIS series of that pair at its location; every tilt-0 pair reads the one
horizontal series. The defaults cover the orientations of residential PV in Europe and the
US (Killinger et al. 2018, [Scenarios](scenarios.md#orientation)).

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.pv_profile.pvgis.orientation.enabled` | boolean | `false` | Draw an orientation per unit and variant. `false` (default): every unit reads the location's one series, and every output is the same as without this section. |
| `scenarios.pv_profile.pvgis.orientation.tilt_deg` | list of integers (each ≥ 0 and ≤ 90), at least 1 item | `[0, 15, 30, 45, 60]` | Slopes of the modules from the horizontal (whole degrees), PVGIS's `angle`. |
| `scenarios.pv_profile.pvgis.orientation.azimuth_deg` | list of integers (each ≥ -180 and ≤ 180), at least 1 item | `[-90, -45, 0, 45, 90]` | Azimuths of the modules (whole degrees, 0 south, 90 west, -90 east), PVGIS's `aspect`. |

### `scenarios.storage_profile`

Storage set-points per scenario ([Scenarios](scenarios.md#storage-set-points)).

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.storage_profile.setpoint` | value spec, ≥ -1 and ≤ 1 | `{distribution: uniform, low: -1.0, high: 1.0}` | Active power of each unit as a fraction of its rating, drawn per unit and scenario: positive discharging, negative charging; no reactive power. Assumption, no source: any operating point within the rating, with no state of charge (a snapshot has none). |

### `scenarios.dg_profile`

Set-points and availability of the dispatchable generators per scenario
([Scenarios](scenarios.md#dispatchable-generators)).

| Key | Type | Default | Description |
|---|---|---|---|
| `scenarios.dg_profile.setpoint` | value spec, ≥ 0 and ≤ 1 | `{distribution: uniform, low: 0.0, high: 1.0}` | Active power of each unit in the power flow as a fraction of its available capacity `a · P^max`, drawn per unit and scenario; no reactive power. Assumption, no source: the source gives the OPF's bounds only, so any point within them. In the OPF the units are dispatched. |
| `scenarios.dg_profile.availability` | value spec, ≥ 0 and ≤ 1 | `1.0` | Factor `a` on each unit's capacity, drawn per unit and scenario: the unit has `a · P^max` available, in the power flow and as the OPF's bound of `p` and `q` (0: out of service, fixed at zero; between 0 and 1: derated). With 1 every unit is available in every scenario; a distribution takes units out, e.g. `{distribution: choice, values: [0, 1], weights: [0.05, 0.95]}` for one unit in twenty out of service. |

## `controls`

Regulator taps and switched-capacitor states per scenario. The state reached is frozen
(`states/`) and the label is the power flow at it with `ControlMode=OFF`. On a network without
control elements both keys are ignored: the state is the compiled one.

| Key | Type | Default | Description |
|---|---|---|---|
| `controls.mode` | one of `static_settle`, `as_is` | `static_settle` | `static_settle`: with the scenario's setpoints, `ControlMode=STATIC` settles regulators and capacitors, the regulated taps are jittered and settle again from there, and the state is frozen; when settling fails, a load ramp reaches the scenario ([Scenarios](scenarios.md)). `as_is`: the taps and capacitor states as compiled. |
| `controls.tap_jitter_steps` | integer ≥ 0 | `2` | Largest jitter `J` (tap steps): each regulated winding's settled tap moves by a uniform integer in [-J, J], within its limits, before the controls settle again; capacitors are not jittered. 0: settle once. Ignored with `as_is`. |

## `labels`

What is solved and stored for each scenario. The power-flow labels are always computed.

| Key | Type | Default | Description |
|---|---|---|---|
| `labels.element_currents` | boolean | `false` | Also store the current of every power-delivery element at each terminal conductor (`scenarios/element_results`); otherwise they follow exactly from the stored data as Yprim · V ([Outputs](outputs.md#6-scenario-level)). |

### `labels.opf`

OPF labels: the minimum-cost OPF of every scenario, verified in OpenDSS
([OPF labels](opf.md)). The power-flow labels stay; the `opf/` tables are added.

| Key | Type | Default | Description |
|---|---|---|---|
| `labels.opf.enabled` | boolean | `false` | Solve and verify the OPF of every scenario. The config is refused unless the `opf` extra (CasADi) is installed. Networks outside the OPF's scope get `not_supported` rows. |
| `labels.opf.costs` | one of `economic`, `sampled` | `economic` | Cost preset. `economic`: source 1 per W, PV units 0 (minimum import), dispatchable generators 0.1 (a tenth of the source's, as Geth, Claeys and Deconinck, PSCC 2022); `sampled`: coefficients drawn per scenario from `sampled`. |
| `labels.opf.time_limit_s` | number > 0 | `300.0` | Wall-clock limit (s) of each Ipopt solve (`max_wall_time`). A scenario not solved from the flat start is solved once more from its power flow, so it can take twice this. Machine-dependent; the 3000-iteration limit is the deterministic bound. |

#### `labels.opf.sampled`

Cost coefficients of the preset ``sampled``, drawn per scenario (``c2 p^2 + c1 p``, ``p``
in W): one pair for the source, one per PV unit, one per dispatchable generator. Ignored
with ``costs: economic``.

| Key | Type | Default | Description |
|---|---|---|---|
| `labels.opf.sampled.source_c1` | value spec, ≥ 0 | `1.0` | Linear cost of the source's power (per W). |
| `labels.opf.sampled.source_c2` | value spec, ≥ 0 | `0.0` | Quadratic cost of the source's power (per W²). |
| `labels.opf.sampled.pv_c1` | value spec, ≥ 0 | `{distribution: uniform, low: 0.0, high: 1.0}` | Linear cost of each PV unit's power (per W), drawn per unit. |
| `labels.opf.sampled.pv_c2` | value spec, ≥ 0 | `0.0` | Quadratic cost of each PV unit's power (per W²). |
| `labels.opf.sampled.dg_c1` | value spec, ≥ 0 | `0.1` | Linear cost of each dispatchable generator's power (per W), drawn per unit after the source's and the PV units' coefficients. Default: a tenth of the source's, as Geth, Claeys and Deconinck (PSCC 2022). |
| `labels.opf.sampled.dg_c2` | value spec, ≥ 0 | `0.0` | Quadratic cost of each dispatchable generator's power (per W²). |

## `views`

Reduced views written next to the full model ([Outputs](outputs.md#7-views)). The
positive-sequence view is not one: ``export-posseq`` computes it from a dataset.

| Key | Type | Default | Description |
|---|---|---|---|
| `views.kron` | boolean | `false` | Kron view (neutral eliminated), re-solved in OpenDSS for every converged scenario (one more solve each; `views/kron/`); ~1 kB per 100 phase nodes and scenario. On a network without an explicit neutral it is only a mapping to the full model. |

## `settings`

How the run is executed and written. `data_dir`, `num_processes`, `contexts_per_process`
and `overwrite` are execution-only: the dataset does not depend on them, they are outside the
config hash, and `generate` overrides them with `--data-dir`, `--processes`, `--contexts` and
`--overwrite`.

| Key | Type | Default | Description |
|---|---|---|---|
| `settings.data_dir` | string of at least 1 character | `./data` | Parent directory of the dataset, `<data_dir>/<dataset_name>`; a relative path is relative to the working directory. |
| `settings.seed` | integer ≥ 0 | required | Root of the seed tree. Every draw is keyed by the seed, the network id, the variant and scenario indices and the quantity, never by the order of work: on one machine the same config, seed and versions give the same tables, but for the measured wall times, for any number of processes and contexts ([Scenarios](scenarios.md), Reproducibility in [OPF labels](opf.md)); a network's draws do not depend on the other networks selected; more variants or scenarios keep the existing ones; changing one quantity's spec does not shift the draws of another. |
| `settings.num_processes` | integer ≥ 1 or null | null | Worker processes; null: one per CPU core (`os.cpu_count()`); 1: in the main process. |
| `settings.contexts_per_process` | integer ≥ 1 | `1` | Threads per process, each with its own OpenDSS context and one chunk of the same variant: more than 1 helps only when a variant has several chunks (`n_scenarios_per_variant` > `chunk_size`). |
| `settings.chunk_size` | integer ≥ 1 | `500` | Scenarios per work unit: one Parquet part per scenario table, the unit of scheduling, resume and failure, held in memory until written. |
| `settings.float_precision` | one of `mixed`, `full` | `mixed` | `mixed`: float32 for currents, powers and setpoints, float64 for voltages and admittances; `full`: float64 everywhere ([Outputs](outputs.md#1-conventions)). |
| `settings.residual_sample_rate` | number ≥ 0 and ≤ 1 | `0.01` | Share of converged scenarios whose full Kirchhoff (KCL) residual is computed and stored in `scenario_meta.residual_rel` (null elsewhere), chosen by a hash of the scenario id; it never changes a status. |
| `settings.overwrite` | boolean | `false` | Delete and regenerate the dataset in `<data_dir>/<dataset_name>` when it has another config hash or versions; `false` refuses. A directory that is not a dataset (no `manifest.json`) is never touched. |
| `settings.validation_tolerance` | number > 0 and ≤ 1e-08 | `1e-10` | OpenDSS `Tolerance` of every solve of the generation, also the solver term of the residual check. At most 1e-8: the setpoint check (1e-6 relative) needs a tolerance 100 times tighter. |
