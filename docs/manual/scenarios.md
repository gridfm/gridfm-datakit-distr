# Scenarios

A *scenario* is one operating point of a variant: loads, PV output, source voltage and the state of
regulators and capacitors, solved in OpenDSS. Scenario `s` of variant `k` of network `i` draws from
its own random streams, so a dataset is the same for any number of processes and any order of
work.

**Loads** (`scenarios.profiles`):

- `synthetic` (default): every nominal load × a global factor per scenario (uniform 0.2–1.2 by
  default) × a lognormal factor per customer (σ 0.3), power factor 0.9–0.99;
- `network_loadshapes`: the source model's own loadshapes at a time drawn per scenario;
- `edp_smart_meter`: EDP's measured profiles.

Loads are constant power within 0.6–1.5 pu: the normalisation widens OpenDSS's window, below
which a load would silently turn into an impedance.

**PV output** (`scenarios.pv_profile`): available power = kWp × a clear-sky shape of the time of
day (sunrise 6:00, noon peak, sunset 18:00) × a cloud factor (Kasten–Czeplak, one cloud cover per
scenario). The time of day is the load profile's when the scenario has one. All of it is injected,
up to the inverter rating (`inverter_kva_per_kwp`; an inverter smaller than its panels clips).
Below unit power factor the inverter absorbs reactive power within its rating.

**Source voltage** (`scenarios.source_voltage_pu`): the magnitude, drawn per scenario (0.98–1.05 pu
by default); balanced, angle 0.

**Controls** (`controls`): with `static_settle` (default) regulators and switched capacitors
settle (OpenDSS `ControlMode=STATIC`); then each regulated tap moves by up to `tap_jitter_steps`
steps, the controls settle again from there, and the state is frozen. A regulator does not move
while its voltage is inside its band, so where it stops depends on where it starts: the jitter
varies the taps across scenarios, and every state is one the controls hold. The label is the power
flow at that fixed state, which the dataset stores (`states/`). With `as_is` the taps and capacitor
states stay as compiled.

Settling starts from the variant's initial taps, and OpenDSS moves a control only after a converged
solve. When the scenario's load cannot be carried at those taps, settling fails twice (the second
time with more iterations) and the scenario is reached by a load ramp instead: every load and PV
setpoint is raised from a tenth of its value to its value in ten steps with the controls active, the
taps carried from step to step, and then the controls settle, jitter and freeze as above. Such
scenarios have `origin` `static_settle+ramp` (or `static_settle+ramp+jitter`). If the ramp fails
too, the scenario is `not_converged` and its `pf_message` says how far the ramp got. A scenario that
settles directly never enters the ramp.

**Labels.** A scenario counts as converged only if every load and generator delivers its
setpoint (within 1e-6 relative); otherwise it is `off_setpoint`. Scenarios that do not converge
keep their metadata row (`pf_status`, `pf_message`) and get no results. A sample of scenarios
(`settings.residual_sample_rate`, 1 % by default) gets the full Kirchhoff residual check
(`residual_rel`).
