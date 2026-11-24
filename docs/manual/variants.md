# Variants

A *variant* is a perturbed copy of a catalog network: its parameters, the phases of some customers
and its rooftop PV and storage units are drawn once, then many [scenarios](scenarios.md) are solved on it.
Variant `k` of network `n` is stored as `n/v{k:04d}` with what was drawn
([Outputs §4](outputs.md#4-variant-level)). Keys: [`variants`](config.md#variants),
[`variants.admittance`](config.md#variantsadmittance),
[`variants.phase_reallocation`](config.md#variantsphase_reallocation), [`der.pv`](config.md#derpv),
[`der.storage`](config.md#derstorage), [`der.dg`](config.md#derdg),
[`variants.upstream`](config.md#variantsupstream).

**Defaults.** 4 variants per network, each with every line length scaled by its own factor
uniform in 0.9–1.1, 20 % of the movable single-phase customers moved to another phase, and PV at a
share of the customers drawn uniformly in 0–0.8; no storage, no dispatchable generators and no
upstream substation. Earthing resistances and neutral impedances stay as in the source. The source network itself is never a variant; this gives one variant equal to it:

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
| Upstream substation | `variants.upstream` | the source of a CSIRO feeder | the transformer rating, hence the source's impedances; the other feeders as loads at the source bus: [below](#upstream-substation) |

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

## Upstream substation

The CSIRO feeders are supplied by an almost ideal LV source (`ISC3 = ISC1 = 100000` A at
0.416 kV), without the MV/LV transformer, the MV network or the other feeders of their
substation, so voltage drop and PV voltage rise are measured from a stiff point.
`variants.upstream.enabled` (off by default) gives every variant its substation. It applies to
the `csiro_enwl` networks only, whose substation's other feeders are in the catalog, and with
the `synthetic` load provider only; `generate` refuses other networks before any work. No
transformer element, node or bus is added: the source keeps its bus, base and role as slack.

- **Rating.** Drawn per variant from the ratings of Electricity North West's distribution
  transformers whose substation has about as many customers as the network (its customers: the
  loads of all its feeders). The table counts 18 133 single-transformer substations of 25 kVA and
  above, from ENWL's open data (CC BY 4.0): the distribution-transformer capacity register joined
  on the substation number with the smart-meter installation counts (`TOTAL CUSTOMERS`). A
  network draws from the substations whose customer count lies in the same octave as its own,
  `[2^k, 2^(k+1))`, the last bin from 512 up; only the counts are in the package
  (`gridfm_datakit_distr.variants.upstream.RATING_COUNTS`). For the 42 to 883 customers of the
  catalog's networks the most frequent ratings are 500 kVA (from 32 to 512 customers) and
  1000 kVA (from 512).
- **Transformer impedance.** The short-circuit impedance `z_k` is the minimum of IEC 60076-5:2006
  Table 1 for the rating: 4.0 % up to 630 kVA, 5.0 % from 631 to 1250 kVA, 6.0 % from 1251 to
  2500 kVA. The resistance is `r = P_k / S_r`, `P_k` the maximum load losses of Regulation (EU)
  548/2014 Annex I Table I.1, Tier 2 (liquid-immersed, from 1 July 2021), linear between its
  rows as the Regulation prescribes; the reactance is `x = sqrt(z_k² - r²)`. For example:

    | Rating (kVA) | `z_k` | `r` | `x` | X/R |
    |---|---|---|---|---|
    | 315 | 4.0 % | 0.89 % | 3.90 % | 4.4 |
    | 500 | 4.0 % | 0.78 % | 3.92 % | 5.0 |
    | 750 | 5.0 % | 0.75 % | 4.94 % | 6.6 |
    | 800 | 5.0 % | 0.75 % | 4.94 % | 6.6 |
    | 1000 | 5.0 % | 0.76 % | 4.94 % | 6.5 |

- **MV network.** 250 MVA at the MV bus, ENWL's design fault level for its 11 kV networks, with
  the transformer's X/R: 0.32 % on the base of an 800 kVA unit, against its 5 %.
- **The source.** The Vsource gets `Z1 = Z2 = Z_T + Z_MV` and `Z0 = Z_T` (Ω at its base voltage):
  a Dyn transformer blocks the MV network's zero sequence. With the neutral bonded to the source's
  star point, as in the CSIRO models, this equals a Dyn transformer fed by the MV network at every
  node, neutrals included, to 3e-8 pu (measured on CSIRO network 13, feeder 1, with the other
  feeders' loads). `variants/source_yprim` holds the resulting admittance; the slack, the bases,
  the source unbalance ([Scenarios](scenarios.md#source-voltage)) and the OPF act on it as on any
  source. There is no transformer element, so no transformer loading among the labels.
- **Other feeders.** Three single-phase loads at the source bus, `Load.upstream_feeders_a`, `_b`
  and `_c`, each from a phase to the neutral (to ground in a model without one). In every scenario
  each draws `g` times the sum of the nominal P and Q of the other feeders' loads on its phase,
  `g` the scenario's global load factor (`scenarios.profiles.synthetic.global_scale`). They are
  not customers: no PV, storage, DG, phase move or per-customer factor, and `der_penetration`
  does not count them. The other feeders are compiled once per dataset in OpenDSS; their names,
  the customer count and the per-phase sums are in `manifest.json` (`networks[].upstream`).
- **Record.** `variants/variants.perturbation`, key `upstream`: `rating_kva`, `z1_ohm` and
  `z0_ohm` (`[R, X]`, Ω) and `loads` (each lumped load's nominal `[P (W), Q (var)]`); the lumped
  loads' set-points are rows of `scenarios/loads` ([Outputs §6](outputs.md#scenariosloads)).

Not modelled: PV and per-customer variation on the other feeders (they follow `g` only), the
transformer's no-load losses and magnetising current, its taps. The DG capacity
([above](#dispatchable-generators)) is computed at the source as loaded.

## Checks and failures

1. **The edits are read back.** A variant is the network compiled with its catalog recipe plus an
   OpenDSS edit script. After compiling it, every line's Yprim must match the drawn perturbation
   within 1e-12 (relative), as must the source admittance against the substation's `Z1` and
   `Z0`, and the earthing resistances, load connections, PV and storage units, dispatchable
   generators and the other feeders' loads must be exactly as drawn. A mismatch is a bug: the variant's units fail with `VariantError`; it is never
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
