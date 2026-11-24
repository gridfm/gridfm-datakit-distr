# OPF labels

With `labels.opf.enabled: true` ([`labels.opf`](config.md#labelsopf)), every scenario of a dataset also gets the solution of its optimal power flow (OPF): the dispatch of the PV units and dispatchable generators that minimises the cost of the power the source and these units deliver, within the voltage limits and the transformer ratings. The power flow of every scenario is still solved and stored, with the uncontrolled dispatch: it sets the OPF's class and its fallback starting point. The OPF tables (`opf/`, [Outputs §8](outputs.md#8-opf)) come next to it.

## Enabling it

Install the `opf` extra (CasADi, which bundles Ipopt: [Installation](../installation.md#optional-extras)) and set in the config:

```yaml
labels:
  opf:
    enabled: true
    costs: economic         # or sampled (labels.opf.sampled)
    time_limit_s: 300       # wall-clock limit of one solve
```

The config is refused when the extra is missing. Each scenario then costs one Ipopt solve (0.2 to 1.5 s on a small CSIRO feeder) and one more OpenDSS power flow, which the dry run's time estimate leaves out. With the OPF off, CasADi is never imported and the dataset has no `opf/` tables. The dataset card (`README.md`) counts the scenarios by OPF status.

## The problem

For one scenario, at the network state, loads and source voltage of its power flow:

- **Controllable:** the source and each PV unit. A PV unit injects `0 ≤ p ≤ p_avail` (its available power in the scenario) and reactive power within its inverter rating, `p² + q² ≤ s_rated²`, and within `|q| ≤ reactive_fraction · s_rated` (default 0.44, IEEE 1547-2018 Category B); no reactive power when its available power is below `reactive_threshold · s_rated` (default 0.05). Both are [`der.pv`](config.md#derpv) value specs, drawn per variant and unit; the limit applied is stored per unit and scenario (`opf_der.q_max_var`). A unit on several phases delivers the same power on each. Each dispatchable generator ([Variants](variants.md#dispatchable-generators), `der.dg`) injects `0 ≤ p ≤ P^max` and `0 ≤ q ≤ P^max`, box bounds without a rating circle, as in the DG case study of Geth, Claeys and Deconinck (PSCC 2022), with `P^max` times the unit's availability in the scenario ([Scenarios](scenarios.md#dispatchable-generators), 1 by default): a unit out of service (availability 0) is fixed at `p = q = 0`, as a PV unit at night; `opf_der.q_max_var` holds the bound.
- **Fixed:** the loads (constant power), the storage units at their set-points ([Scenarios](scenarios.md#storage-set-points); no reactive power), the source voltage, taps and capacitor steps. Storage is not a control here: in a single snapshot its stored energy has no value to trade against, so it becomes one only with time sequences of scenarios.
- **Constraints:** Kirchhoff's current law on the network admittance the dataset stores (`Y_state`, [Outputs §5](outputs.md#5-network-state-level): lines, switches, transformers at their taps, capacitors, reactors) with the source as an ideal voltage behind its impedance; loads, PV units and dispatchable generators connected wye or delta, on one to three phases; each phase-to-neutral voltage magnitude (phase-to-ground at a bus without a neutral) within [`networks.voltage_limits`](config.md#networksvoltage_limits); the apparent power into winding 1 of every transformer within its rating (`variants/elements.rating_va`). Line ampacities are not constraints yet.
- **Objective:** `Σ (c2 p² + c1 p)` over the source, the PV units and the dispatchable generators, `p` the active power each delivers. The coefficients are stored with every scenario (`opf_meta.source_c1`, `source_c2`, `opf_der.c1`, `c2`, with the source's power in `opf_meta.source_p_w`). Two cost presets (`labels.opf.costs`): `economic` gives the source `c1 = 1` per W and the PV units 0, the minimum import, that is the largest use of the available PV power net of losses, and each dispatchable generator `c1 = 0.1` per W, ten times less than the source as in the paper; `sampled` draws the coefficients per scenario from [`labels.opf.sampled`](config.md#labelsopfsampled) (by default the source at 1 per W, each PV unit uniform in 0 to 1 per W, each dispatchable generator 0.1 per W), so that the dispatch depends on the costs, as datakit's cost perturbation does for transmission. The source's power is signed: export to the upstream grid counts as negative import, so cheaper dispatchable generators raise their output until a limit is active and may export, as in the paper, where the upstream grid is "an unbounded generator [that] models the import/export". The costs are linear only: fuel-cost models (as PGLib's) would need an import price that no source gives for a distribution feeder.

The problem is the unbalanced AC OPF in current-voltage form with an explicit neutral, written on the node voltages in rectangular coordinates, and solved with Ipopt (through CasADi) from a flat start, at a tolerance of 1e-9; when that start does not end `Solve_Succeeded`, the scenario is solved again from its power-flow solution, a feasible point when the scenario is of class `A` (`opf_meta.opf_start`). It is scaled to per unit on each node's base voltage and on a power base set by the network's loads (1 kVA on LV feeders, 100 kVA to 1 MVA on MV feeders). Ipopt returns local optima: no label claims global optimality.

**Voltage limits.** `vmin_pu` and `vmax_pu` are per unit of each bus's nominal line-to-neutral voltage (`variants/nodes.kv_nominal_ln`, else the base), as in the statistics. The catalog declares a nominal voltage where it differs from the base ([Networks](networks.md#how-a-network-is-prepared)): the CSIRO feeders and the European LV feeder are nominally 230 V phase-to-neutral on a 416 V base, so the defaults 0.94 and 1.10 (preset `uk_esqcr`) give the UK limits of 216.2 to 253.0 V.

**Verification.** A solved dispatch is set on the PV units and dispatchable generators in OpenDSS, with the storage units still at their set-points, and the power flow is solved with the controls off. The label is `verified` when every load and generator delivers its set point, every node voltage equals the OPF's within 1e-6 per unit of its base, and no phase-to-neutral voltage exceeds its bounds by more than 1e-6 per unit. On a network with regulators or switched capacitors, the label is the optimum at the scenario's taps and steps; one more solve with the controls on tells whether they would hold that state at the solved dispatch (`opf_meta.controls_hold`). The limits active at the solution are counted (`n_vmin_active`, `n_vmax_active`, `n_rating_active`).

## Scope

The OPF covers networks of lines, switches, transformers, capacitors and reactors fed by one voltage source, with loads, PV units, storage units and dispatchable generators connected wye or delta on one to three phases: every network of the catalog but IEEE 37. Regulators and switched capacitors are held at the scenario's state. In a zone that only OpenDSS's anti-float admittance of the transformer windings ties to ground (bus 610 of IEEE 123, behind a delta-delta transformer), the voltage to ground is a numerical convention: there the bounds and the verification are on the phase-to-phase voltages, at the same per-unit limits of the nominal phase-to-phase voltage and the same 1e-6 pu, and the nodes are marked `ground_by_convention` in `variants/nodes`. A network with a zone tied to ground only through line capacitance (IEEE 37 behind its delta-delta substation), where the OPF's convergence is not established, or with a PV system, an OpenDSS `Storage` element or a generator other than the PV units, storage units and dispatchable generators, keeps its power-flow labels and gets the status `not_supported` with the reason, per scenario (`opf_meta.opf_message`) and once in `manifest.json` (`networks[].opf_not_supported`).

## Statuses

`opf_meta.opf_status`, one row per scenario, nothing dropped:

| Status | Meaning |
|---|---|
| `Solve_Succeeded` | solved: the label (check `verified` too) |
| another Ipopt status | e.g. `Infeasible_Problem_Detected` (no dispatch meets the limits: a violation the PV units and dispatchable generators cannot remove), `Maximum_WallTime_Exceeded` (`labels.opf.time_limit_s`), `Solved_To_Acceptable_Level`: not a label |
| `not_supported` | the network is outside the scope above |
| `no_pf_label` | the scenario's power flow gave no label (not converged, or off its set points); not solved |
| `no_free_variable` | class B with every PV unit fixed (none placed, or no available power and no reactive range) and no dispatchable generator in service (none placed, or all with availability 0): nothing can move to remove the violation, e.g. at night when charging storage pulls the voltages below their limits; not solved. A class-A scenario in that case is solved as before (its label is the power flow) |
| `error` | building, solving or verifying raised; the message is in `opf_message` |

`opf_class` says what the OPF had to do: `A` when the uncontrolled power flow already keeps the voltages within the limits and the transformers within their ratings (with `economic` the OPF then keeps the PV at its available power and moves the reactive power to reduce the losses), `B` when it does not, `C` without a power-flow label.

A class-A scenario has a feasible point by construction (its power flow), yet a few end unsolved:
in the campaign on the default configuration, 9 of about 6 900 class-A scenarios ended infeasible, at
an acceptable level or at the iteration limit, all on hard MV networks (EDP rural, IEEE 123, IEEE
8500). They are recorded with their status and carry no OPF label; the stiff admittance of the
voltage source is the main cause found so far.

## Reproducibility

The general rules are in [Scenarios](scenarios.md#seeds-and-reproducibility). On one machine
the OPF labels are the same whatever the number of processes or the thread count of the linear
algebra (OpenBLAS at 1 or 6 threads: identical voltages and dispatches, measured on two machines);
`opf_meta.t_solve_s` is a wall time and differs.

Across machines the numbers agree within the solvers' accuracy, not bit for bit. Measured between
a desktop (Intel i5-9400F) and a cluster node (Intel Xeon Gold 6530) with identical library
versions: OPF dispatches that both machines solve within 2e-8 of the largest voltage, 1e-5 W in the
objective and 3e-7 W per PV unit, well inside the 1e-6 pu verification. Rare borderline OPF solves end with a different status: on 3
of 11 hard EDP rural scenarios one machine returned `Solve_Succeeded` and the other
`Infeasible_Problem_Detected` or `Solved_To_Acceptable_Level`. Such rows are recorded with their
status and never become labels unless solved and verified, so a regenerated dataset can differ
from the original in which of those scenarios carry OPF labels.

`labels.opf.time_limit_s` is a wall-clock limit, so a solve that ends with
`Maximum_WallTime_Exceeded` depends on the machine and its load. Such scenarios carry that status
and no label. The iteration limit (3000 Ipopt iterations, not configurable) is the deterministic
bound.
