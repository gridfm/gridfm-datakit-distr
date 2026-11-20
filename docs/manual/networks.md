# Network catalog

The networks a dataset can use, where they come from, how they are prepared and whether they pass
the conformance suite. Sources and checksums: `gridfm_datakit_distr/networks/catalog.yaml` and
`gridfm_datakit_distr/networks/manifests/`. Status with DSS C-API 0.14.5.

## Sources and the cache

| Source | Networks | Origin | Licence |
|---|---|---|---|
| `epri_ieee` | `ieee13`, `ieee34`, `ieee37`, `ieee123`, `ieee8500`, `ieee_eulv` | OpenDSS models of the IEEE PES test feeders and of the IEEE European LV feeder (DSS-Extensions mirror of the EPRI distribution) | feeder data without formal licence; scripts BSD-3-Clause |
| `csiro_enwl` | `csiro_enwl/nNN_fM`: 128 four-wire LV feeders of the ENWL networks | CSIRO data collection, DOI 10.25919/jaae-vc35 | CC BY 4.0; cable impedance data CC BY-NC 3.0 |
| `edp` | `edp/industrial`, `edp/rural`, `edp/urban` | Mendeley Data, DOI 10.17632/gspyzvvrhm.2 (RAR5 archives) | CC BY 4.0 |

Nothing of them is in the repository: [`fetch`](cli.md#fetch) downloads a source into the network
cache and verifies every file against its SHA-256 manifest. Every network carries its licence and
citation in its passport and in the dataset card.

## How a network is prepared

Every command that uses a network (`inspect`, `validate`, `generate`) prepares it the same way:

1. **Compile.** OpenDSS compiles the master file with the network's catalog recipe: an optional cut
   of the master before a given line, documented correction commands, base loads. The model is read
   back from the engine, never parsed; disabled elements are left out.
2. **Normalise loads.** Constant P and Q between 0.6 and 1.5 pu
   ([`normalisation.loads`](config.md#normalisationloads)): below OpenDSS's default 0.95 pu a load
   would silently become an impedance.
3. **Voltage bases.** Checked against the equipment ratings; buses on a wrong level are re-based.
   Where the catalog declares a nominal voltage different from the base, voltage limits and
   statistics use it: the CSIRO feeders and the European LV feeder are nominally 230 V phase to
   neutral on a 416 V line-to-line base.
4. **Base case.** Solved with the controls off and the taps as compiled; its convergence,
   Kirchhoff residual and setpoints are reported.

`inspect <id>` prints the result; `inspect <id> --passport-dir DIR` writes the network's passport
([Outputs §3](outputs.md#3-network-level)): provenance, licence, recipe, normalisation, known
differences.

## Selecting networks

[`networks.include`](config.md#networks) takes catalog ids or `fnmatch` patterns
(`csiro_enwl/n01_*`; `*` is the whole catalog). Networks are ordered by id; that order numbers them
in `scenario_id`.

Some networks have other **representations** of the same feeder, selected with `@`:

| Network | Representations (default first) |
|---|---|
| `ieee8500` | the balanced load case; `unbalanced` |
| CSIRO feeders | `four_wire`; `kron` (Kron-reduced), `phase_to_neutral`, `modified_phase_to_neutral` (three-wire) |

`csiro_enwl/n01_*@kron` selects one; `networks.neutral_representation: kron` selects `kron`
wherever it exists. `explicit` (default) and `as_is` both load each network's default model; the IEEE
models have no neutral conductor, and `generate` notes it. The CSIRO three-wire representations
and IEEE 8500's unbalanced case are not yet validated.

## Conformance suite

`validate` ([command](cli.md#validate)) runs on the prepared network:

| Check | Passes when |
|---|---|
| C1 | the base case converges |
| C2 | the base-case solution satisfies Kirchhoff's current law on the extracted network admittance, within the solver's tolerance |
| C3 | the network admittance assembled from the elements' Yprims equals OpenDSS's system admittance without the loads' and sources' contributions (1e-9 relative) |
| C4 | every enabled element is of a supported class |
| C5 | phase labels are consistent and no node is islanded |
| C6 | the neutral representation is the one the catalog declares |
| C7 | the structure and base-case voltages match the golden file (only with `--golden`) |
| bases | voltage bases consistent with the equipment |
| setpoints | every load and generator delivers its setpoint in the base case |

`generate` does not run the suite: it requires every variant's base case to pass C1 and C2
([Variants](variants.md#checks-and-failures)). A network that fails is not left out automatically.

## Networks

| Network | Source | Buses / nodes | Loads | Neutral | Levels (kV LL) | Recipe | Conformance |
|---|---|---|---|---|---|---|---|
| `ieee13` | `epri_ieee` | 16 / 41 | 15 | kron | 115, 4.16, 0.48 | — | C1–C7 pass |
| `ieee34` | `epri_ieee` | 37 / 95 | 68 | kron | 69, 24.9, 4.16 | six 832–858 delta loads connected phase to phase | C1–C7 pass |
| `ieee37` | `epri_ieee` | 39 / 117 | 30 | kron | 230, 4.8, 0.48 | — | C1–C7 pass |
| `ieee123` | `epri_ieee` | 132 / 278 | 91 | kron | 4.16, 0.48 | — | C1–C7 pass |
| `ieee_eulv` | `epri_ieee` | 907 / 2721 | 55 | kron | 11, 0.416 | — | C1–C7 pass |
| `ieee8500` | `epri_ieee` | 4876 / 8531 | 1177 (split phase) | kron | 115, 12.47, 0.208 | — | C1–C7 pass |
| `csiro_enwl/nNN_fM` (128 four-wire feeders) | `csiro_enwl` | 236 (n23_f3) – 28 092 (n08_f2) nodes | 2–302 | explicit | 0.416 | bases re-based from the 0.48 kV defaults | 127 of 128 pass C1–C6 (`n13_f4` fails C1, below); C7 on the 26 feeders of `integrationtests/golden/networks.txt` |
| `edp/industrial` | `edp` | 777 / 2888 | 191 | mixed (MV Kron, LV four-wire) | 22, 0.4 | master cut before the monitors, loadshape-mean base loads, LV neutrals grounded through 5 + j0.01 Ω | C1–C7 pass |
| `edp/rural` | `edp` | 4649 / 18 595 | 2731 | explicit | 22, 0.4 | master cut, loadshape-mean base loads | C1–C7 pass |
| `edp/urban` | `edp` | 6444 / 25 775 | 35 297 | explicit | 22, 0.4 | master cut, loadshape-mean base loads | C1–C7 pass |

Voltage levels list the zone nominal voltages: regulated zones whose nominal comes from a
regulator winding (2.4 kV × √3 = 4.157 kV) are shown under their system level. The European LV and
EDP base loads are placeholders in the sources (1 kW per load); EDP's are replaced by their
loadshape means, the European LV's keep their own loadshapes. Only these networks (EDP and the
European LV feeder) carry loadshapes ([Scenarios](scenarios.md#loads)).

## Known differences and exclusions

- **IEEE 34:** the published `ieee34Mod1.dss` ties six delta loads to ground; the corrected model
  differs from it by up to 2.1e-3 pu.
- **Base cases with compiled taps:** labels use the control states settled per scenario as
  inputs (`controls`, [Scenarios](scenarios.md#controls)). With the taps as compiled, IEEE 34 and
  IEEE 8500 reach 0.70 pu at the feeder end and IEEE 37 0.87 pu.
- **CSIRO `n13_f4`:** at its nominal loads (148 kW on 785 buses) OpenDSS converges only up to 34 %
  of them under constant power; the published model converges because its loads become constant
  impedances below 0.95 pu. It stays in the catalog: a pattern such as `csiro_enwl/*` selects it,
  and then every unit of its variants fails. Leave it out of `networks.include`.
- **PMD's own parser** differs from OpenDSS on these sources (mile = 1609.3 m, dropped grounding
  reactors), so the conformance suite has no cross-engine check (C8). Engine agreement is checked
  through the OPF labels: each dispatch is verified in OpenDSS ([OPF labels](opf.md)), and the OPF
  is cross-checked against PMD on small CSIRO feeders (`integrationtests/test_opf_pmd.py`, Julia).
