# gridfm-datakit-distr

**gridfm-datakit-distr** generates machine-learning datasets of power flows in **three-phase
unbalanced distribution networks**: MV and LV, including European four-wire LV feeders with an
explicit neutral. It is the distribution counterpart of
[gridfm-datakit](https://github.com/gridfm/gridfm-datakit).

- **The truth, per conductor.** OpenDSS solves the full three-phase model. Every node (phases and
  neutral) gets its voltage, and the admittance of every element and of the network is stored, so
  currents, flows and injections follow exactly.
- **Real networks.** A catalog of IEEE test feeders, 128 four-wire ENWL feeders (CSIRO) and EDP
  MV/LV networks, downloaded with checksums and checked by a conformance suite.
- **Variety.** Variants (line lengths, neutral impedances, earthing, phases of customers, rooftop
  PV) and scenarios (load profiles, PV output, source voltage, regulator and capacitor states).
- **Physics-checked labels.** A scenario counts as converged only if every load and generator
  delivers its setpoint, and a sample gets the full Kirchhoff residual check. Failures are
  recorded, never dropped.
- **Reproducible.** The same config and seed give the same bytes, whatever the number of
  processes.
- **gridfm-graphkit today.** `export-posseq` writes the positive-sequence equivalent of a dataset in
  gridfm-datakit's format.

Start with [Installation](installation.md) and [Getting started](manual/getting_started.md).
