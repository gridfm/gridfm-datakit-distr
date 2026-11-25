# Scenarios

A *scenario* is one operating point of a [variant](variants.md): load setpoints, PV output,
storage and dispatchable-generator set-points, source voltage and the state of regulators and
capacitors, solved in OpenDSS. Keys:
[`scenarios`](config.md#scenarios), [`scenarios.profiles`](config.md#scenariosprofiles),
[`scenarios.pv_profile`](config.md#scenariospv_profile), [`controls`](config.md#controls),
[`labels`](config.md#labels), [`views`](config.md#views).

Each scenario, in order:

1. draws its loads, PV output, storage and DG set-points, source voltage and source unbalance;
2. sets them in OpenDSS;
3. resets the controls to the variant's compiled state, then settles and freezes them, or keeps
   them ([controls](#controls));
4. solves the power flow with the controls frozen (`ControlMode=OFF`);
5. checks that every load and generator delivers its setpoint and, on a sample, the Kirchhoff
   residual;
6. stores its row in `scenarios/scenario_meta`, its setpoints and, if converged, its voltages.

## Loads

The customers are the enabled loads of the variant. `scenarios.profiles.provider` decides their
setpoints:

| Provider | Setpoints |
|---|---|
| `synthetic` (default) | `P = L · g · m · P_nominal`: `L` per variant (`variant_scale`, log-uniform on [1/2.64, 2.64], below), `g` per scenario (`global_scale`, uniform 0.1–1.2), `m` per customer, lognormal with mean 1 (`exp(σz − σ²/2)`, `σ = local_sigma` = 0.3), so the mean load is `L · E[g] · P_nominal`. `Q = P · tan(acos(pf))`, lagging, with `pf` drawn per customer (`power_factor`, uniform 0.9–0.99): the source's nominal Q is not used. No time of day. |
| `network_loadshapes` | the loads' own daily (else yearly) loadshapes, as OpenDSS reads them. A time is drawn uniformly over the profile window (the length of the shortest multi-point shape): one for all customers (`same_timestamp: true`, default, coincident demand) or one per customer (`false`). Without a Q curve, Q keeps the load's kvar/kW ratio. A customer with a one-point shape keeps that value, one without a shape its nominal value. |
| `edp_smart_meter` | the same reader: in the EDP networks the loadshapes are the smart-meter files. The name is what `scenario_meta.profile_source` records. |

**Default load scale.** The CSIRO feeders' nominal loads (about 0.9 kW per customer) behave like an
after-diversity design peak, not a typical demand: with `L = 1`, `g = 1` is the design peak. The
figures of this paragraph are for `g` alone (`L = 1`); the load level below spreads the networks
around them. The default
`global_scale`, uniform 0.1–1.2, is chosen to cover the states measured today and plausible heavier
ones, not to reproduce today's distribution. Against Electricity North West's measured substation
loads (open data, `lv_load_duration`, CC BY 4.0: a year of half-hourly load at each secondary
substation; per customer, on the about 6 200 substations with at least 20 customers): at the
lightest 5 % of substations a year's 10th percentile is about 0.14 of the nominal or less, which the
floor 0.1 covers; the ceiling 1.2 lies above a year's 90th percentile at 86 % of the substations and
leaves room for more electric heating and vehicles. The density is not matched to today: drawn
uniformly, a dataset holds more heavy states than a year measured now; narrow the range for today's
typical level. On every network the nominal loads stand for a peak, so `g = 1` means the same
thing everywhere: the IEEE test cases' published loads (the peak by OpenDSS's convention for base
kW; the files do not say), the European LV feeder's placeholders (55 kW, 0.96 of its profiles'
coincident peak over the day) and the EDP networks' coincident peak over their profile window
([Networks](networks.md#networks)).

**Load level per network.** `g` moves a network's demand in time against its own peak; it cannot
say that one network's customers consume more than another's. They do: the customers of one
feeder share building type, era and heating, so a network's level has a common part, and an
independent factor per customer (`m`) averages out over a feeder instead of producing it (with
`σ = 0.3`, about 8 % on a feeder of 14 customers). Measured, the peak load per customer varies
widely across networks: the 99th percentile of the 15-minute load per
housing unit of 24 residential LV feeders of FeederBW (Netze BW, Zenodo
10.5281/zenodo.17831177, CC BY 4.0) runs from 0.44 to 2.99 kW, and the 90th percentile of the
half-hourly load per customer at Electricity North West's substations from 0.43 to 2.10 kW (5–95 %
range), while at `L = 1` the middle half of the catalog's CSIRO feeders lies within 1.10–1.23 kW
on the FeederBW index. Each variant therefore draws a load level `L`
(`scenarios.profiles.synthetic.variant_scale`) that multiplies every nominal load of the variant,
the other feeders' loads of the [substation](variants.md#upstream-substation) included, and is
recorded as `variant_scale` in `variants/variants.perturbation`. A variant stands for a network of
the same topology with other customers. The default, log-uniform on [1/k, k] with k = 2.64, has
median 1 (half the variants below their nominal level, half above) and mean
(k − 1/k) / (2 ln k) = 1.165, so the mean load is about 16 % above that of `L = 1`; summed over
the feeders of a substation, whose levels are drawn independently, the 90th percentile of the
load per customer rises by about a fifth (measured on the CSIRO networks without PV: median 0.98
against 0.80 kW). k is the widest of the measured ratios to the CSIRO feeders' median at `L = 1`:
0.38–2.58 (FeederBW) and 0.52–2.53 (Electricity North West), rounded to two decimals.
`variant_scale: 1` turns it off.

- The networks with catalog `load_level: false` keep `L = 1`; nothing is drawn for them
  ([Networks](networks.md#load-level)): the EDP networks, whose nominal load is the measured
  coincident peak, and the IEEE MV feeders (13, 34, 37, 123, 8500), whose loads are spot or
  aggregate loads over many substation areas while `L` is measured on LV feeders of one area
  each. A loadshape provider draws none either.
- `L` does not enter the variant's checks (the base case at the nominal loads), the dispatchable
  generators' capacity or the substation's rating: they stay as without it, so heavier networks
  meet the same transformer and DG as lighter ones.
- Why not a wider `g` instead: `g` drawn over a wider range gives every network the same spread of
  states around its own peak, so the networks stay alike, and it stretches `g`, time against the
  network's own peak, beyond 1.2 times that peak.

Only the EDP networks and the IEEE European LV feeder (`ieee_eulv`: one day at 1-minute steps)
carry loadshapes. `generate`, with or without `--dry-run`, refuses a loadshape provider on any
other network before any work (exit 2), naming the networks. No source dates its profiles:
`profile_time_s` counts seconds from the start of the profile. `scenarios.profiles.synthetic` may
be set only with `provider: synthetic`.

**Load model.** Every load draws constant P and Q (`normalisation.loads.model: constant_power`,
the only model) between `vminpu` and `vmaxpu` (0.6 and 1.5 pu): OpenDSS's default window is
widened because outside it a load silently becomes a constant impedance (the scenario would then
be `off_setpoint`). The stored setpoints therefore determine the voltages
([Outputs](outputs.md#scenariosloads)).

## PV output

For each PV unit ([Variants](variants.md#rooftop-pv)), per scenario, the available power is
`p_avail = kWp · (output per kWp) · a`. Two providers give the output per kWp
(`scenarios.pv_profile.provider`): a synthetic clear sky (`synthetic_clear_sky`, the default) and
hourly series of PVGIS (`pvgis`).

**Against measured penetration.** The share of customers with PV
([`der.pv.penetration`](variants.md#rooftop-pv)) is drawn per variant uniform on [0, 0.8) by
default, median 0.40. Electricity North West's 14 374 distribution substations with 5 or more
customers have a median of 0.043 MCS-certified PV installations per customer (SP Electricity North
West open data, MCS installations per substation), a lower bound since MCS certifies only part of
the installations: the default's median is about 9 times it, and 32.5 % of its draws lie within
the substations' 5–95 % range (0.003–0.263). A default dataset therefore describes mostly networks
with far more PV than today's. Hence frequent reverse flow: at the default config the load
of most of the catalog's CSIRO networks, all feeders summed as at their substation, is negative at
its 10th percentile over scenarios,
against 0.7 % of the substations Electricity North West monitors (SP Electricity North West open
data, LV load duration curves). The dataset card states this when the penetration is the default.

### Synthetic clear sky

- available power `p_avail = kWp · clear_sky(h) · (1 − 0.75 c^3.4) · a`;
- `clear_sky(h) = max(0, cos(π (h − 12) / 12))`: sunrise 6:00, noon peak, sunset 18:00, the same
  every day, no location or date;
- `c`, the cloud cover (`scenarios.pv_profile.cloudiness`, uniform 0–1), one per scenario for the
  whole network; `1 − 0.75 c^3.4` is the Kasten–Czeplak relation for global irradiance;
- `a`, the unit's availability (`scenarios.pv_profile.availability`), drawn per unit and scenario
  independently: shading, or 0 for a unit out of service (not orientation, which would shift a
  unit's daily shape the same way in every scenario). The default 1 gives every unit of a
  variant the same share of its kWp in a scenario, so their outputs are proportional across
  scenarios; a distribution breaks that, e.g. `{distribution: choice, values: [0, 1], weights:
  [0.05, 0.95]}` puts one unit in twenty out of service;
- `h`, the time of day of the load profile when all customers share one time (the profile's start
  taken as midnight). Otherwise (`synthetic` loads, or `same_timestamp: false`) `h` is uniform
  over 24 h, independent of the loads: half of such scenarios fall at night and have no PV
  output.

**Against measured weather.** The synthetic sky has the sun of an equinox overhead at noon. At
Manchester (53.48° N; PVGIS 5.3, SARAH-3, 2005–2023, PVGIS's optimal angles, 14 % system loss) a
1 kWp system gives on average 0.32 kW in the local noon hour and 0.10 kW over all hours, against
0.83 kW and 0.26 kW here: at that latitude the synthetic provider overstates PV output about
2.6 times.

### PVGIS series

`provider: pvgis` takes the output per kWp from PVGIS 5.3, the photovoltaic geographical
information system of the EU Joint Research Centre: for each location, one hourly series of the
power `P` (W) of a 1 kWp system over the configured years (`scenarios.pv_profile.pvgis`, by
default 2005–2023), computed by PVGIS from irradiance data (PVGIS-SARAH3 where it covers the site, as in
Europe; ERA5 elsewhere, e.g. in the US) on modules at PVGIS's optimal fixed slope and azimuth, with its temperature model and
14 % system loss. Then `p_avail = kWp · P / 1 kW · a`, with `a` the unit's availability as above;
no cloud cover is drawn (`cloudiness` must keep its default).

- **Locations.** No source gives a point for any catalog network
  ([Networks](networks.md#locations)), so none is built in: `scenarios.pv_profile.pvgis.locations`
  maps network ids or glob patterns to a latitude, a longitude and an IANA time zone, and
  `generate` refuses (exit 2, before any work, also with `--dry-run`) a selected network without
  one. The sample `scripts/config/pvgis.yaml` places the CSIRO feeders at Manchester.
- **Download.** `fetch --config CONFIG` downloads one series per location (with
  [`orientation`](#orientation), one per orientation class) into the cache (`<cache>/pvgis/`) and records the file's SHA-256 next to it; every later use checks the file
  against it, and a changed file is an error (`fetch --config CONFIG --force` downloads it
  again). `generate` never downloads: without the series it stops and names the command.
  Rate-limited or overloaded answers (HTTP 429, 529), dropped connections and timeouts are
  retried three times.
- **Time.** PVGIS stamps hourly records in UTC (inferred: `seriescalc` does not say), SARAH3 ones
  about ten minutes past the hour; each record is taken as the hour that starts at its `HH:00`.
  A scenario draws one record for the whole network from its own stream. With a load time of
  day `h` (the same case in which the clear sky uses the profile's time), a date is drawn
  uniformly among the local days the series covers entirely, and the record is the one holding
  the local time `h` on that date (Python's `zoneinfo`, so daylight saving applies; a wall time
  that does not exist, or occurs twice, takes the offset in force before the change). Without
  one, the record is drawn uniformly among all hours of the series, night included. The hour
  drawn is stored as `scenario_meta.pv_time_utc_s`.
- **Attribution.** The data are free to use if the source is acknowledged: "PVGIS © European
  Communities, 2001-2026". Each network's request, the databases and angles PVGIS used and the
  series' SHA-256 are recorded in `manifest.json` (`networks[].pvgis`) and in its passport
  (`pv_series`), with `orientation` one record per class; the dataset card carries the
  attribution.
- **Limits.** One point per network and, without `orientation`, one system orientation for
  every unit; the series of
  a location is the same for all its networks, so two feeders placed at one point see the same
  weather in the same hour. The date drawn is independent of the load profile's (no catalog
  profile is dated), so only the time of day couples PV output and loads.

### Orientation

`scenarios.pv_profile.pvgis.orientation.enabled: true` gives each PV unit its own orientation,
so the units of one variant peak at different hours. Per variant, each unit draws one (tilt,
azimuth) pair from its own stream, uniformly among the pairs of `tilt_deg` and `azimuth_deg`
(whole degrees; azimuth 0 south, 90 west, −90 east, PVGIS's convention). A unit keeps its pair
at every higher penetration, as it keeps its size. Each pair is one PVGIS series per location,
requested with that `angle` and `aspect`: PVGIS's own transposition to the module plane,
temperature model and losses apply, and nothing is transposed here. Every tilt-0 pair reads the
same horizontal series (requested with azimuth 0). In a scenario all units read their series at
the same record, so they share the hour and the weather. Each unit's pair is recorded in
`variants/variants.perturbation` (`pv_orientation`, `[tilt, azimuth]`).

- **Classes.** The defaults, tilt {0, 15, 30, 45, 60}° × azimuth {−90, −45, 0, 45, 90}°, are 25
  pairs and 21 series per location; `fetch --config` downloads them all (about 16 MB each for
  19 years). They cover the orientations of residential PV that Killinger et al. fitted to the
  metadata of 2.8 million systems in 12 regional clusters (Solar Energy 173, 2018,
  doi:10.1016/j.solener.2018.08.051). Over the European clusters and USA North and South, the
  fits' 10th to 90th percentiles span tilt 9.0°–48.7° and azimuth −61.2°–72.3°, inside the
  classes' 0°–60° and −90°–90°. The classes cover that range; they do not follow the measured
  density, which concentrates around south.
- **PVGIS only.** The synthetic clear sky has the geometry of latitude 0 at an equinox: there a
  module tilted toward the equator by `β` delivers `cos β` times the horizontal output at every
  hour, with no shift of the peak. At the networks' latitudes the same tilt raises the yield, so
  the clear sky would get the sign wrong: `orientation` with `synthetic_clear_sky` is a config
  error. It also needs `optimalangles: false` and no `angle` or `aspect`, since it sets the
  angles itself.
- **Limits.** Discrete classes, not a continuous distribution. Killinger's fits miss the spike
  of systems facing due south. No published distribution was found for Portugal or Spain: the
  EDP networks are covered by the European and US clusters.

### Inverter

The unit injects `p = min(p_avail, s_rated)` (an inverter smaller than its panels clips). The
power flow never curtails: that is the OPF's job ([OPF labels](opf.md)). Its reactive power
follows `der.pv.reactive_mode`:

- `power_factor` (default): `q = −p · tan(acos(pf))`, limited to `p² + q² ≤ s_rated²`. At the
  default power factor 1, `q = 0`.
- `uniform`: `q = u · q_lim`, with `u` drawn per unit and scenario, uniform on [−1, 1), and
  `q_lim = min(q_max, √(s_rated² − p²))`. `q_max` is the OPF's limit: `reactive_fraction · s_rated`, and
  0 when `p_avail < reactive_threshold · s_rated` (0.44 and 0.05 by default, IEEE 1547-2018
  Category B; [OPF labels](opf.md#the-problem)). Positive `q` is injected, negative absorbed. This
  is IEEE 1547-2018's constant reactive power mode, with its setpoint drawn per scenario within
  the limits the OPF applies, so the power flow and the OPF share one reactive range.
  `power_factor` must stay 1.

Limit of `uniform`: `q` does not follow the voltage. An inverter in IEEE 1547-2018's volt-var
mode injects only below the zero point of its curve and absorbs only above it; the draw also
gives reactive power injected at high voltage, absorbed at low voltage, and inside a volt-var
deadband, states that volt-var never produces. Voltage-dependent control (volt-var, volt-watt) is
not modelled.

## Storage set-points

Each storage unit ([Variants](variants.md#storage)) is fixed at `p = setpoint · kW` per scenario,
with `setpoint` (`scenarios.storage_profile.setpoint`, uniform in −1 to 1 by default) drawn per
unit and scenario: positive discharging into the network, negative charging; `q = 0`. A snapshot
has no state of charge, so any set-point within the rating is admitted, at any hour; the unit is
fixed in the power flow and in the OPF ([OPF labels](opf.md#the-problem)). In `scenarios/der` its
rows have `der_type` `bess` and `p_avail_w` equal to `p_w`.

## Dispatchable generators

Each dispatchable generator ([Variants](variants.md#dispatchable-generators)) is set in the power
flow to `p = setpoint · a · P^max`, with `setpoint` (`scenarios.dg_profile.setpoint`, uniform in 0
to 1 by default) drawn per unit and scenario, and `q = 0`. This operating point is an assumption:
the source of the DG setup gives only the OPF's bounds, in which the OPF then dispatches the units
([OPF labels](opf.md#the-problem)).

`a`, the unit's availability (`scenarios.dg_profile.availability`), is drawn per unit and
scenario: the unit has `a · P^max` available, in the power flow and as the OPF's bound of `p` and
`q`. A unit with `a = 0` is out of service, at `p = q = 0` in both; between 0 and 1 it is derated.
The default 1 keeps every unit available in every scenario; a distribution takes units out, e.g.
`{distribution: choice, values: [0, 1], weights: [0.05, 0.95]}` puts one unit in twenty out of
service. No outage rate is assumed by default.

In `scenarios/der` the rows have `der_type` `dg`, `p_avail_w` equal to `a · P^max` and
`s_rated_va` equal to `P^max`.

## Source voltage

`scenarios.source_voltage_pu`: the magnitude of the variant's voltage source, drawn per scenario
(uniform 0.98–1.05 pu), angle 0; with a negative sequence it is the magnitude of the positive one.

`scenarios.source_unbalance`: the negative-sequence ratio `u2` of the source's ideal voltage, drawn
per scenario (default 0: balanced, and nothing below applies). The angle of the negative sequence
relative to the positive one is uniform in [0, 2π) per scenario: no source prefers an angle, and
the standards treat `u2` as a ratio of magnitudes. The scenario's ideal source voltages are then
`E_s = source_v_pu · (E + u2 · exp(j θ) · N)`, `E` the source's voltages at 1 pu and `N` their
negative sequence ([Outputs §4](outputs.md#variantssourcepart-parquet)). There is no zero
sequence: IEC TR 61000-3-13 defines no compatibility level for it, and the descriptions of EN 50160
we found (secondary sources) bound the negative sequence only.

`u2` is the ratio behind the source impedance, not at the source bus: there the drop of the source
current across the source impedance, itself unbalanced by single-phase loads, moves the ratio by up
to about 2e-4 on the stiff CSIRO sources and 5e-5 on IEEE 13 (at 0.5–2 % drawn; at no load the bus
has the drawn ratio). With the substation's impedance
([`variants.upstream`](variants.md#upstream-substation)) the offset grows: up to 6.6e-3 on CSIRO
network 13, feeder 1, at 0–2 % drawn, whose other feeders load one phase most. The stored `source_u2` and `source_u2_angle_rad` and the source tables give
the source exactly, as the power flow and the OPF see it.

For coverage of the background unbalance that real supply voltages carry from upstream, set
`source_unbalance: {distribution: uniform, low: 0, high: 0.02}`. EN 50160 bounds the 10-minute
negative-sequence ratio at 2 % for 95 % of a week, and the CIGRE/CIRED C4.103 survey found about
95 % of MV and LV sites below 2 % at their weekly 95th percentile; both figures come from secondary
sources (the standard's text was not available to us). The uniform distribution is a coverage
choice, not a measured one: no source gives the distribution of `u2` over time.

In OpenDSS the negative sequence is a current source (`Isource`, `sequence=neg`) at the voltage
source's nodes that injects `Y_s E_2`: with the voltage source it acts as one source `E_1 + E_2`
behind the source impedance, and it adds no node. The extracted model, its node and element tables
and `variants/source` are the same as with a balanced source; the Kron view gets the same source.
It needs a three-phase voltage source: with a non-zero `source_unbalance`, every variant of a
network without one fails, with the reason in `manifest.json`. The positive-sequence view
([`export-posseq`](outputs.md#72-positive-sequence-export-export-posseq-not-stored-in-the-dataset)) keeps its source at `|V1|`, the positive sequence of the full
model's source-bus voltages, and so ignores the negative sequence by construction.

## Controls

`controls.mode: static_settle` (default): with the scenario's setpoints applied, the regulators and
switched capacitors settle within one solve (`ControlMode=STATIC`). Then each regulated tap moves
by a uniform integer number of steps in `[−J, J]` (`J = tap_jitter_steps`, default 2), clipped to
its limits, and the controls settle again from there. A regulator does not move while its voltage
is inside its band, so where it stops depends on where it starts: the jitter varies the taps across
scenarios, and every state is one the controls hold. The state is then frozen and the label is the
power flow at that state (`ControlMode=OFF`). Capacitor steps are not jittered. `as_is`: taps and
capacitor steps as compiled. On a network without control elements nothing settles.

States are stored once per variant, by content (`states/`, `state_id`); a scenario records its
`state_id` and how it got there (`origin`: `static_settle`, `static_settle+jitter` when the jitter
led to another state, `as_is`, and the `+ramp` forms below).

**Load ramp.** Settling starts from the variant's initial taps, and OpenDSS moves a control only
after a converged solve. When the scenario's load cannot be carried at those taps, settling fails
twice (the second time with five times the iterations) and the scenario is reached by a load ramp:
every load and PV setpoint is raised from a tenth of its value to its value in ten steps with the
controls active, the taps carried from step to step; then the controls settle, jitter and freeze as
above (`origin` `static_settle+ramp` or `static_settle+ramp+jitter`). If the ramp fails too, the
scenario is `not_converged` and `pf_message` says how far the ramp got. A scenario that settles
directly never enters the ramp.

**Not a time series.** Each scenario's taps are settled from the compiled state, not carried over
from an earlier time. A real regulator keeps its tap while the voltage stays in its band, so its
position depends on how the load got there. Compared with a quasi-static time series over a week
of hourly household profiles on IEEE 13 and IEEE 34, the default (`J = 2`) visits every tap
position the time series visits, but with different frequencies, and the per node-phase voltage
quantiles (1 %, 50 %, 99 %) differ by one to two tap steps (0.006 to 0.012 pu). Every label is the
exact power flow at its state; the mix of states is not the one a feeder goes through over time.
Only networks with regulators are affected.

## Power flow and statuses

Every solve starts from the same point (`Set Mode=Snap`), with OpenDSS `Tolerance` =
`settings.validation_tolerance` (1e-10) and at most 200 iterations; a solve that does not converge
is retried once with 1000. Every scenario gets a row in `scenarios/scenario_meta`, whatever
happens:

| `pf_status` | Meaning | Voltages stored |
|---|---|---|
| `converged` | converged, and every load and generator delivers its setpoint within 1e-6 (relative) | yes |
| `off_setpoint` | converged, but `n_off_setpoint` loads or generators do not deliver their setpoint (for instance a load outside its voltage window) | no |
| `not_converged` | no converged solution; `pf_message` reports the load ramp if it was tried | no |
| `error` | an exception; `pf_message` holds its type and message | no |

Setpoints (`scenarios/loads`, `scenarios/der`) are stored for every scenario whose draws were made.
A sample of scenarios (`settings.residual_sample_rate`, 1 %, chosen by a hash of the scenario id)
gets the full Kirchhoff residual check of the stored solution; its value is recorded in
`residual_rel` and does not change `pf_status`.

**Optional outputs.** `labels.element_currents: true` also stores every power-delivery element's
current at each terminal conductor (`scenarios/element_results`; otherwise they are recomputed
exactly as Yprim · V). `views.kron: true` re-solves every converged scenario in OpenDSS on the
variant with the neutral eliminated (Kron-reduced lines, neutrals merged into ground, wye windings
solidly grounded), with the same setpoints and network state, and stores that solution with a
`view_status` per scenario; on a network without neutral conductors the view equals the full model
and is stored as a mapping only ([Outputs §7.1](outputs.md#71-kron-view-viewskron)).

## Seeds and reproducibility

There is no global random state. Every draw comes from a `numpy.random.SeedSequence` fixed by
`settings.seed`, the network's id (via its SHA-256), the variant index, the scenario index and the
quantity drawn (loads, PV, source voltage and unbalance, tap jitter, …). Hence:

- a scenario depends only on the seed, its network's id and its indices: not on the other networks
  of the dataset, on the number of variants or scenarios, or on the order of the work. A dataset
  with more variants or scenarios and the same seed repeats the draws and solutions of a smaller
  one; between datasets with different network lists a network's data differ only in the network
  bits of `scenario_id` and in `config_hash`;
- each quantity has its own stream: changing the spec of one (say the cloud cover) leaves the draws
  of the others unchanged.

**On one machine**, the same config, seed and library versions give the same tables value for value
whatever the number of processes and contexts, except the measured wall times (`t_solve_s`,
`t_extract_s`, and `opf_meta.t_solve_s`); `manifest.json` also differs in its execution settings and
peak memory. **Across machines** with identical library versions, the power-flow voltages agree
within 1e-9 pu (measured between a desktop and a cluster node), not bit for bit; OPF labels:
[OPF reproducibility](opf.md#reproducibility).
