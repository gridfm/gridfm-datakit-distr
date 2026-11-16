# OPF labels

With `labels.opf.enabled: true`, every scenario of a dataset also gets the solution of its optimal power flow (OPF): the dispatch of the PV units that minimises the cost of the power the source and the PV units deliver, within the voltage limits. Each scenario keeps its power-flow label, the uncontrolled dispatch; the OPF tables (`opf/`, [Outputs §8](outputs.md#8-opf)) come next to it.

## The problem

For one scenario, at the network state, loads and source voltage of its power flow:

- **Controllable:** the source and each PV unit. A PV unit injects `0 ≤ p ≤ p_avail` (its available power in the scenario) and any reactive power `q` within its inverter rating, `p² + q² ≤ s_rated²`.
- **Fixed:** the loads (constant power), the source voltage, taps and capacitor steps.
- **Constraints:** Kirchhoff's current law on the network admittance the dataset stores (`states/ybus`) with the source as an ideal voltage behind its impedance; each phase-to-neutral voltage magnitude (phase-to-ground at a bus without a neutral) within `networks.voltage_limits`.
- **Objective:** `Σ (c2 p² + c1 p)` over the source and the PV units, `p` the active power each delivers. The coefficients are stored with every scenario (`opf_meta.source_c1`, `source_c2`, `opf_der.c1`, `c2`). The only cost preset so far, `economic`, gives the source `c1 = 1` per W and the PV units 0: the minimum import, that is the largest use of the available PV power net of losses.

The problem is the unbalanced AC OPF in current-voltage form with an explicit neutral, written on the node voltages in rectangular coordinates, and solved with Ipopt (through CasADi) from a flat start, at a tolerance of 1e-9. Ipopt returns local optima: no label claims global optimality.

**Voltage limits.** `vmin_pu` and `vmax_pu` are per unit of the line-to-neutral base voltage of the network's source bus (`variants/nodes.kv_base_ln`), the base the statistics use. On the CSIRO feeders that base is 240.2 V, so the defaults 0.94 and 1.10 give 225.8 to 264.2 V; the UK limits of 216.2 to 253.0 V are `vmin_pu: 0.9002`, `vmax_pu: 1.0534` there.

**Verification.** A solved dispatch is set on the PV units in OpenDSS and the power flow is solved with the controls off. The label is `verified` when every load and generator delivers its set point, every node voltage equals the OPF's within 1e-6 per unit of its base, and no phase-to-neutral voltage exceeds its bounds by more than 1e-6 per unit.

## Scope

The OPF covers networks made of lines, a neutral-grounding reactor and one voltage source, with single-phase phase-to-neutral loads and PV units: the CSIRO four-wire LV feeders. A network with anything else (a transformer, a capacitor, a regulator, a switch, a delta or multi-phase load) keeps its power-flow labels and gets the status `not_supported` with the reason, per scenario (`opf_meta.opf_message`) and once in `manifest.json` (`networks[].opf_not_supported`).

## Statuses

`opf_meta.opf_status`, one row per scenario, nothing dropped:

| Status | Meaning |
|---|---|
| `Solve_Succeeded` | solved: the label (check `verified` too) |
| another Ipopt status | e.g. `Infeasible_Problem_Detected` (no dispatch meets the limits: a violation the PV units cannot remove), `Maximum_WallTime_Exceeded` (`labels.opf.time_limit_s`), `Solved_To_Acceptable_Level`: not a label |
| `not_supported` | the network is outside the scope above |
| `no_pf_label` | the scenario's power flow gave no label (not converged, or off its set points); not solved |
| `error` | building, solving or verifying raised; the message is in `opf_message` |

`opf_class` says what the OPF had to do: `A` when the uncontrolled power flow already keeps the voltages within the limits (the OPF then keeps the PV at its available power and moves the reactive power to reduce the losses), `B` when it does not, `C` without a power-flow label.

## Enabling it

Install the `opf` extra (CasADi, which bundles Ipopt):

```bash
pip install "gridfm-datakit-distr[opf] @ git+ssh://git@github.com/gridfm/gridfm-datakit-distr.git"
```

and set in the config:

```yaml
normalisation:
  loads:
    model: constant_power   # the default; the OPF needs it
labels:
  opf:
    enabled: true
    costs: economic
    time_limit_s: 300       # wall-clock limit of one solve
```

The config is refused when the extra is missing or the loads are not constant power. Each scenario then costs one Ipopt solve (0.2 to 1.5 s on a small CSIRO feeder) and one more OpenDSS power flow; the labels do not depend on the number of processes. With the OPF off, CasADi is never imported and the dataset has no `opf/` tables.
