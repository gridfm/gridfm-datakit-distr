# Variants

A *variant* is a perturbed copy of a catalog network: its parameters, the phases of some customers
and its rooftop PV units are drawn once, then many [scenarios](scenarios.md) are solved on it.
Variant `k` of network `n` is stored as `n/v{k:04d}` with what was drawn
([Outputs §4](outputs.md#4-variant-level)). Keys: [`variants`](config.md#variants),
[`variants.admittance`](config.md#variantsadmittance),
[`variants.phase_reallocation`](config.md#variantsphase_reallocation), [`der.pv`](config.md#derpv).

**Defaults.** 4 variants per network, each with every line length scaled by its own factor
uniform in 0.9–1.1, 20 % of the movable single-phase customers moved to another phase, and PV at a
share of the customers drawn uniformly in 0–0.8. Earthing resistances and neutral impedances stay
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

## Checks and failures

1. **The edits are read back.** A variant is the network compiled with its catalog recipe plus an
   OpenDSS edit script. After compiling it, every line's Yprim must match the drawn perturbation
   within 1e-12 (relative), and the earthing resistances, load connections and PV units must be
   exactly as drawn. A mismatch is a bug: the variant's units fail with `VariantError`; it is never
   redrawn.
2. **The base case must hold.** With its nominal loads and no PV output, the variant must converge
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
