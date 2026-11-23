# Scenarios

A *scenario* is one operating point of a [variant](variants.md): load setpoints, PV output,
source voltage and the state of regulators and capacitors, solved in OpenDSS. Keys:
[`scenarios`](config.md#scenarios), [`scenarios.profiles`](config.md#scenariosprofiles),
[`scenarios.pv_profile`](config.md#scenariospv_profile), [`controls`](config.md#controls),
[`labels`](config.md#labels), [`views`](config.md#views).

Each scenario, in order:

1. draws its loads, PV output and source voltage;
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
| `synthetic` (default) | `P = g · m · P_nominal`: `g` per scenario (`global_scale`, uniform 0.2–1.2), `m` per customer, lognormal with mean 1 (`exp(σz − σ²/2)`, `σ = local_sigma` = 0.3), so the mean load is `E[g] · P_nominal`. `Q = P · tan(acos(pf))`, lagging, with `pf` drawn per customer (`power_factor`, uniform 0.9–0.99): the source's nominal Q is not used. No time of day. |
| `network_loadshapes` | the loads' own daily (else yearly) loadshapes, as OpenDSS reads them. A time is drawn uniformly over the profile window (the length of the shortest multi-point shape): one for all customers (`same_timestamp: true`, default, coincident demand) or one per customer (`false`). Without a Q curve, Q keeps the load's kvar/kW ratio. A customer with a one-point shape keeps that value, one without a shape its nominal value. |
| `edp_smart_meter` | the same reader: in the EDP networks the loadshapes are the smart-meter files. The name is what `scenario_meta.profile_source` records. |

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

For each PV unit ([Variants](variants.md#rooftop-pv)), per scenario:

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

The unit injects `p = min(p_avail, s_rated)` (an inverter smaller than its panels clips) and
`q = −p · tan(acos(pf))`, limited to `p² + q² ≤ s_rated²`. The power flow never curtails: that is
the OPF's job ([OPF labels](opf.md)).

## Source voltage

`scenarios.source_voltage_pu`: the magnitude of the variant's voltage source, drawn per scenario
(uniform 0.98–1.05 pu); balanced, angle 0.

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
quantity drawn (loads, PV, source voltage, tap jitter, …). Hence:

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
