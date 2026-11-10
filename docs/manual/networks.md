# Network catalog

The networks the generator can use, where they come from, how they are compiled and whether they
pass the conformance suite. Sources and checksums: `gridfm_datakit_distr/networks/catalog.yaml`
and `gridfm_datakit_distr/networks/manifests/`. Status with DSS C-API 0.14.5.

## How a network enters a dataset

1. `fetch <source>` downloads it into the cache and checks every file against the manifest.
2. It is compiled by OpenDSS with its catalog recipe (`networks/compile.py`): an optional cut of
   the master, documented correction commands, base loads.
3. It is normalised: constant-power loads with a 0.6–1.5 pu window, voltage bases from the
   equipment ratings.
4. `validate` runs the conformance suite: C1 base case solves, C2 KCL residual, C3
   network admittance against the solver's, C4 element coverage, C5 phase consistency, C6 neutral
   representation, C7 golden file, plus voltage bases and base-case setpoints.
   `inspect <id> --passport-dir` writes the network's passport ([outputs](outputs.md), §3).

Networks that fail the suite are not used.

## Networks

| Network | Source | Licence | Buses / nodes | Loads | Neutral | Levels (kV LL) | Recipe | Conformance | v0.1 |
|---|---|---|---|---|---|---|---|---|---|
| `ieee13` | `epri_ieee` | IEEE PES test feeder; scripts BSD-3-Clause | 16 / 41 | 15 | kron | 115, 4.16, 0.48 | — | C1–C7 pass | yes |
| `ieee34` | `epri_ieee` | as above | 37 / 95 | 68 | kron | 69, 24.9, 4.16 | six 832–858 delta loads connected phase to phase | C1–C7 pass | yes |
| `ieee37` | `epri_ieee` | as above | 39 / 117 | 30 | kron | 230, 4.8, 0.48 | — | C1–C7 pass | yes |
| `ieee123` | `epri_ieee` | as above | 132 / 278 | 91 | kron | 4.16, 0.48 | — | C1–C7 pass | yes |
| `ieee_eulv` | `epri_ieee` | as above | 907 / 2721 | 55 | kron | 11, 0.416 | — | C1–C7 pass | yes |
| `ieee8500` | `epri_ieee` | as above | 4876 / 8531 | 1177 (split phase) | kron | 115, 12.47, 0.208 | — | C1–C7 pass | yes |
| `csiro_enwl/nNN_fM` (128 four-wire feeders) | `csiro_enwl` | CC BY 4.0; cable impedance data CC BY-NC 3.0 | 236 (n23_f3) – 28 092 (n08_f2) nodes | 2–302 | explicit | 0.416 | bases re-based from the 0.48 kV defaults | 127 of 128 pass C1–C6; C7 on the 26 feeders of `integrationtests/golden/networks.txt` | 127 feeders; `n13_f4` excluded |
| `edp/industrial` | `edp` | CC BY 4.0 | 777 / 2888 | 191 | mixed (MV Kron, LV four-wire) | 22, 0.4 | master cut before the monitors, loadshape-mean base loads, LV neutrals grounded through 5 + j0.01 Ω | C1–C7 pass | yes |
| `edp/rural` | `edp` | CC BY 4.0 | 4649 / 18 595 | 2731 | explicit | 22, 0.4 | master cut, loadshape-mean base loads | C1–C7 pass | yes |
| `edp/urban` | `edp` | CC BY 4.0 | 6444 / 25 775 | 35 297 | explicit | 22, 0.4 | master cut, loadshape-mean base loads | C1–C7 pass | yes |

Voltage levels list the zone nominal voltages: regulated zones whose nominal comes from a
regulator winding (2.4 kV × √3 = 4.157 kV) are shown under their system level. The European LV and
EDP base loads are placeholders in the sources (1 kW per load); EDP's are replaced by their
loadshape means, the European LV's keep their own loadshapes (profile provider `network_loadshapes`).

## Known differences and exclusions

- **IEEE 34:** the published `ieee34Mod1.dss` ties six delta loads to ground; the corrected model
  differs from it by up to 2.1e-3 pu.
- **Base cases with compiled taps:** labels use ControlMode OFF with control states as inputs
 . With the taps as compiled, IEEE 34 and IEEE 8500 reach 0.70 pu at the feeder end and
  IEEE 37 0.87 pu; datasets settle the control states per scenario (`controls`).
- **CSIRO `n13_f4`:** at its nominal loads (148 kW on 785 buses) OpenDSS converges only up to 34 %
  of them under constant power; the published model converges because its loads become constant
  impedances below 0.95 pu. Excluded from v0.1.
- **PMD's own parser** differs from OpenDSS on these sources (mile = 1609.3 m, dropped grounding
  reactors): PMD inputs come from our semantic layer; engine agreement (C8) is
  checked with the OPF labels (next version).
- **Not yet validated:** the CSIRO Kron and phase-to-neutral representations, the unbalanced-load
  representation of IEEE 8500 (selected through configuration).
