# Variants

A *variant* is a perturbed copy of a catalog network: its parameters, the phases of some customers
and its rooftop PV and storage units are drawn once, then many [scenarios](scenarios.md) are solved on it.
Variant `k` of network `n` is stored as `n/v{k:04d}` with what was drawn
([Outputs §4](outputs.md#4-variant-level)). Keys: [`variants`](config.md#variants),
[`variants.admittance`](config.md#variantsadmittance),
[`variants.phase_reallocation`](config.md#variantsphase_reallocation), [`der.pv`](config.md#derpv),
[`der.storage`](config.md#derstorage), [`der.dg`](config.md#derdg).

**Defaults.** 4 variants per network, each with every line length scaled by its own factor
uniform in 0.9–1.1, 20 % of the movable single-phase customers moved to another phase, and PV at a
share of the customers drawn uniformly in 0–0.8; no storage and no dispatchable generators. Earthing resistances and neutral impedances stay
as in the source. The source network itself is never a variant; this gives one variant equal to it:

```yaml
variants:
  n_variants: 1
  admittance: {line_length_scale: 1.0}
  phase_reallocation: {enabled: false}
der:
  pv: {penetration: 0}
```

## Perturbations

Every value is drawn independently per element (a value spec gives its distribution; a number
fixes it).

| Perturbation | Key | Elements | What changes |
|---|---|---|---|
| Line length | `variants.admittance.line_length_scale` | every enabled line that is not a switch | the length × a factor `a`: series impedance and shunt admittance scale by exactly `a` |
| Neutral impedance | `variants.admittance.neutral_impedance_scale` | lines with neutral conductors (models with an explicit neutral) | the neutral self impedances × `β`, the phase-neutral mutual impedances × `√β`, so the line stays passive for any `β > 0`; capacitances unchanged |
| Earthing resistance | `variants.admittance.grounding_resistance_ohm` | existing earth electrodes: wye transformer windings grounded through `Rneut ≥ 0`, grounding reactors (neutral to ground) | their resistance (Ω); `null` keeps the source's |
| Phase reallocation | `variants.phase_reallocation` | enabled single-phase loads with one phase conductor, at a bus that has another phase | `fraction` of them (rounded), chosen uniformly, each moved to one of the other phases of its bus, chosen uniformly; its neutral connection is kept |
| Rooftop PV | `der.pv` | every enabled load (a *customer*) | PV units: [below](#rooftop-pv) |
| Storage | `der.storage` | the customers with PV | storage units: [below](#storage) |
| Dispatchable generators | `der.dg` | every 4th customer | not drawn: the same units in every variant, [below](#dispatchable-generators) |

Not perturbed: switches (impedance and state; topology never changes), a grounding reactor that
bonds the neutral to a source's star point (CSIRO's `Reactor.grounding`, which carries the whole
return current), neutrals tied solidly to node 0 in a bus definition, phase-to-phase and
multi-phase loads. A perturbation without elements to act on does nothing, and in the catalog:

- neutral impedances exist only on four-wire lines: the CSIRO feeders and the EDP networks;
- earth electrodes exist only in the EDP networks (the wye windings of their LV transformers,
  grounded through `Rneut`). The IEEE feeders have none, and the CSIRO feeders' only grounding
  element is the neutral's bond at the source bus: on both, `grounding_resistance_ohm` changes
  nothing;
- IEEE 37, IEEE 8500 and EDP industrial have no movable single-phase loads.

## Rooftop PV

- **Placement.** The penetration `ρ` (`der.pv.penetration`) is drawn once per variant; each customer
  then gets PV when its own uniform draw is below `ρ`, so the realised share
  (`scenario_meta.der_penetration`) scatters around `ρ`. Placement is monotone in the penetration:
  a customer with PV at 20 % has it, with the same size, at 40 %. `penetration: 0` places none.
- **Unit.** An OpenDSS generator `Generator.pv_<load>` on its customer's nodes (after any phase
  move), with the customer's phases, voltage and wye or delta connection; constant P and Q within
  the load voltage window ([`normalisation.loads`](config.md#normalisationloads)).
- **Size and inverter.** `kwp` per unit; inverter rating `inverter_kva_per_kwp` × kWp (1.1 kVA per
  kWp by default); `power_factor` per unit (1 by default; below 1 the inverter absorbs reactive
  power within its rating).
- **OPF only.** `reactive_fraction` and `reactive_threshold`, per unit, bound the reactive power the
  OPF may dispatch ([OPF labels](opf.md#the-problem)).

The output of each unit is drawn per scenario ([Scenarios](scenarios.md#pv-output)).

## Storage

- **Placement.** The share (`der.storage.share`, 0 by default: no storage) is drawn once per
  variant; each customer with PV then gets a storage unit when its own uniform draw is below it.
  As for PV, placement is monotone in the share, with the same rating at every share.
- **Unit.** An OpenDSS generator `Generator.bess_<load>` on its customer's nodes, connected as its
  PV unit, constant P within the load voltage window, with a signed set-point per scenario
  ([Scenarios](scenarios.md#storage-set-points)). Not an OpenDSS `Storage` element: that element
  idles itself silently when its stored energy reaches a limit, which a snapshot cannot track.
- **Rating.** By default (`kw` null) from the peak power of the customer's PV unit, by the rule for
  customer batteries of NREL's SMART-DS, a synthetic dataset of US distribution networks
  ([documentation](https://github.com/openEDI/documentation/blob/main/SMART-DS/Readme.md),
  Appendix 1): below 4 kWp 4 kW, from 4 to 10 kWp 8 kW, from 10 to 150 kWp 25 kW, from 150 kWp
  100 kW. The source gives the bands of installed PV capacity (kW) as `<4`, `4-10`, `10-150` and
  `>150`; that each lower bound belongs to the band above it is our reading. With the default PV sizes (2–6 kWp) a unit
  is 4 or 8 kW. SMART-DS sizes the stored energy at twice the rating (kWh = 2 × kW); a snapshot
  has no state of charge, so it is not used. A value spec for `kw` draws the rating per unit and
  variant instead, e.g. `{distribution: uniform, low: 3.0, high: 5.0}`. The rating is stored as
  the unit's `s_rated_va`.

## Dispatchable generators

The DG case study of Geth, Claeys and Deconinck ("Optimal power flow in four-wire distribution
networks: formulation and benchmarking", PSCC 2022, §IV-B): "a single-phase DG at every 4th
load", every DG with the same capacity `P^max`, "sized such that [...] the DGs can congest the
network", with `0 ≤ p ≤ P^max` and `0 ≤ q ≤ P^max` in the OPF. `der.dg.enabled` (off by default)
places them in every variant of every network. What the paper does not fix is ours:

- **Sites.** The 4th, 8th, 12th, ... enabled load in name order (a string sort: `h1`, `h10`, `h2`,
  ...), the order of the customers everywhere in this package; the paper gives no order and no
  start. A unit takes its load's phases, so it is single-phase where the load is.
- **Unit.** An OpenDSS generator `Generator.dg_<load>` on its customer's nodes (after any phase
  move), connected as the load, constant P within the load voltage window, set per scenario
  ([Scenarios](scenarios.md#dispatchable-generators)).
- **Capacity.** One `P^max` per network, computed once per dataset: the smallest equal capacity at
  which the units congest the unperturbed network, at its nominal loads, without PV and storage,
  every unit at `p = P^max`, `q = 0`, controls off (the taps as compiled), solved in OpenDSS.
  *Congest*: a bounded voltage of the OPF ([OPF labels](opf.md#the-problem): phase to neutral,
  `networks.voltage_limits`) reaches the upper bound, or a transformer reaches its rating at
  winding 1 (the OPF's only branch limit), counting only transformers below their rating without
  DG. A scan of `P = s · 2^k`, `k = 0 … 6` (`s`: the customers' total nominal apparent power per
  unit) brackets the first point at a limit, and a bisection narrows it to 0.1 % of `P`; the scan
  is needed because a transformer's flow first falls, then rises with `P`. A step that jumps over
  the limit to a point where the power flow diverges (EDP urban) is bisected in turn, so the limit
  is still found.
- **Record.** `manifest.json` (`networks[].dg`): `status`, `n_units`, `p_max_w` (W), `binding`
  (`voltage` or the transformer's name) and `ignored` (transformers above their rating without
  DG); the dataset card lists them, and `variants/variants.perturbation` holds `dg_kw` per load.
  A network without a capacity gets no DG, with its status: `no_site` (fewer than 4 loads),
  `voltage_at_bound` (a voltage at or above the upper bound without DG), `no_binding_limit`
  (nothing reached at the largest `P` of the scan), `not_converged` (the power flow diverges
  before a limit is reached) or `off_setpoint`.

## Checks and failures

1. **The edits are read back.** A variant is the network compiled with its catalog recipe plus an
   OpenDSS edit script. After compiling it, every line's Yprim must match the drawn perturbation
   within 1e-12 (relative), and the earthing resistances, load connections, PV and storage units
   and dispatchable generators must be exactly as drawn. A mismatch is a bug: the variant's units fail with `VariantError`; it is never
   redrawn.
2. **The base case must hold.** With its nominal loads and no DER output, the variant must converge
   (C1) and pass the Kirchhoff residual check (C2) ([conformance suite](networks.md#conformance-suite)).
   If not, the whole variant is drawn again from fresh random streams: three attempts in all. The
   accepted attempt is recorded (`attempt` in `variants/variants.perturbation`). After three
   failures every unit of the variant is `failed` in `manifest.json` with
   `variant failed: attempt 0: …; attempt 1: …; attempt 2: …`, and the variant has no scenarios.

## Seeds

Variant `k` depends only on `settings.seed`, the network's id and `k`: not on the other networks
of the dataset, on `n_variants` or on the order of the work. Each perturbation draws from its own
random stream, so changing one key does not change what the others draw (unless the base-case
check then rejects an attempt). Details: [seeds and reproducibility](scenarios.md#seeds-and-reproducibility).
