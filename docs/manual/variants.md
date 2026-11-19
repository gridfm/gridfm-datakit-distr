# Variants

A *variant* is a perturbed copy of a catalog network: its structure changes once, then many
scenarios are solved on it. Variant `k` of a network draws from its own random streams of
`settings.seed`, so it is the same whatever else the dataset contains. The defaults draw 4
variants per network, each with line lengths scaled uniformly in 0.9–1.1, a share of single-phase
customers moved to another phase, and PV at a share of customers drawn uniformly in 0–0.8. With
`n_variants: 1`, `line_length_scale: 1.0`, `phase_reallocation: {enabled: false}` and
`der.pv.penetration: 0` the variant is the source network.

| Perturbation | Config | What changes |
|---|---|---|
| Line length | `variants.admittance.line_length_scale` | one factor per line: its series impedance and shunt admittance scale with it; switches keep their impedance |
| Neutral impedance | `variants.admittance.neutral_impedance_scale` | on four-wire lines, the neutral's self impedance × β and the phase-neutral mutual impedances × √β, so the line stays passive |
| Earthing resistance | `variants.admittance.grounding_resistance_ohm` | the resistance of the existing earth electrodes (`Rneut` of grounded wye windings, grounding reactors); the neutral bond at a source is kept |
| Phase reallocation | `variants.phase_reallocation` | a share of the single-phase customers (`fraction`) move to another phase of their bus, chosen uniformly, with their neutral; phase-to-phase loads stay |
| Rooftop PV | `der.pv` | which customers have PV (`penetration`), its size (`kwp`) and power factor; a unit sits on its customer's nodes, after any phase move |

PV placement is monotone in the penetration: one uniform draw per customer decides it, so a
customer with PV at 20 % has it, with the same size, at 40 %.

**Checks.** Each variant is applied to OpenDSS as an edit script, then read back and compared with
what was drawn (line admittances to 1e-12). A mismatch is an error and is never redrawn. The
variant's base case must converge and pass the Kirchhoff residual check. If it does not, the
variant is redrawn from the next attempt's streams, three times at most, and then recorded as
failed.
