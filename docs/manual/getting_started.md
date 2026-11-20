# Getting started

From an empty machine to a dataset of three-phase unbalanced power flows read in Python, then OPF
labels and larger runs. The test suite (`integrationtests/test_tutorial.py`) runs the IEEE 13
commands of sections 2 and 4, every Python snippet (its output must match the one shown) and the
OPF example, so they work as written.

**The model in one paragraph.** A catalog network ([Networks](networks.md)) is perturbed into
*variants*: line lengths, earthing resistances, phases of customers, rooftop PV
([Variants](variants.md)). Each variant is solved in OpenDSS for many *scenarios*: loads, PV
output, source voltage, regulator and capacitor states ([Scenarios](scenarios.md)). Every node
(each phase and neutral conductor of each bus) of every converged scenario gets its complex
voltage, and the admittances of the network and of every element are stored, so currents, flows
and injections follow from the voltages (section 4). Simplified versions are derived from this
truth: the Kron view (neutral eliminated, `views.kron`) and the positive-sequence equivalent in
gridfm-datakit's format (`export-posseq`).

## 1. Install and fetch the networks

You need Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). The guide works in a clone of the
repository, for its sample configs in `scripts/config/` (pip and the optional extras:
[Installation](../installation.md)).

```bash
git clone https://github.com/gridfm/gridfm-datakit-distr.git && cd gridfm-datakit-distr
uv sync
uv run gridfm_datakit_distr fetch ieee13 csiro_enwl   # IEEE feeders, CSIRO LV feeders
uv run gridfm_datakit_distr inspect ieee13            # optional: the network's summary
```

`fetch ieee13` installs the whole IEEE source; every file is checked against a SHA-256 manifest
and cached outside the repository (location, options and the EDP archives:
[`fetch`](cli.md#fetch); sources and licences: [Networks](networks.md#sources-and-the-cache)).

## 2. A first dataset: IEEE 13 in a few seconds

`scripts/config/ieee13_small.yaml` asks for 2 variants × 20 scenarios of IEEE 13. Estimate first:

```bash
uv run gridfm_datakit_distr generate scripts/config/ieee13_small.yaml --dry-run
```

The dry run prepares the network and prints the number of scenarios and of work units, the nodes
and customers of each network, and estimates of time and storage; it writes nothing. Then
generate and summarise:

```bash
uv run gridfm_datakit_distr generate scripts/config/ieee13_small.yaml --data-dir data
uv run gridfm_datakit_distr stats data/ieee13_small
```

`generate` prints the units by status, the scenarios by power-flow status, the solve time and the
size on disk, and exits with 1 if a unit failed (section 7). Its note `no explicit neutral in
ieee13` says that the source model is three-wire (Kron-reduced): its nodes are phase conductors
only. `stats` prints and writes to `stats/stats.json` the violation rates and the distributions of
phase and neutral voltages, voltage unbalance and loadings (`--plots`, with the `plots` extra, also
draws them).

Running the same `generate` again **resumes**: finished units are kept, the others run, so an
interrupted run loses nothing. A changed config or version is refused unless you pass
`--overwrite`, which deletes the old dataset ([rules](cli.md#generate)).

## 3. What was written

```text
data/ieee13_small/
  manifest.json     resolved config, its hash, versions, every work unit's status
  README.md         dataset card: networks, licences, counts by status
  croissant.json    MLCommons Croissant 1.0 metadata of every table
  catalog/          the networks and their passports (provenance, licence, recipe)
  variants/         per variant: nodes, elements, terminals, admittances, source, semantic layer
  states/           per network state (taps, capacitor steps): the admittances it changes
  scenarios/        per scenario: status, node voltages, load and PV setpoints
  stats/            written by `stats`
```

Each piece of data is stored once, at the level where it changes: network, variant, network state
(regulator taps and capacitor steps), scenario. Tables are Parquet: a directory of parts, one per
work unit (`scenarios/node_results/part-*.parquet`), or a single file
(`states/ybus.parquet`). Every table and column, units and conventions: [Outputs](outputs.md).

Nothing is dropped. Every scenario has a row in `scenarios/scenario_meta` with its `pf_status`
(`converged`, `off_setpoint`, `not_converged`, `error`) and a message; only `converged` scenarios
have voltages ([statuses](scenarios.md#power-flow-and-statuses)). A work unit that fails as a
whole is marked `failed` in `manifest.json` with its message.

## 4. Reading the data

`pyarrow.parquet.read_table` reads a directory of parts and a single file alike. The snippets use
the IEEE 13 dataset of section 2 and run in sequence.

```python
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET = Path("data/ieee13_small")
meta = pq.read_table(DATASET / "scenarios" / "scenario_meta")
print(meta.num_rows, "scenarios:", pc.value_counts(meta["pf_status"]).to_pylist())
```

Output:

```text
40 scenarios: [{'values': 'converged', 'counts': 40}]
```

Each scenario has a `scenario_id`, its `variant_id`, the `state_id` of its network state and a
`pf_status`.

### Voltages of a scenario

Nodes are the conductors of the buses (`a`, `b`, `c`, `n`, …) of a variant, numbered by
`node_idx`. The label of a power-flow model is one complex voltage per node, to ground, in volts,
as two float columns. The three phases of bus 671 below differ in magnitude and are not exactly
120° apart: the network is unbalanced.

```python
first = meta.filter(pc.equal(meta["pf_status"], "converged")).slice(0, 1).to_pylist()[0]
sid, variant, state = first["scenario_id"], first["variant_id"], first["state_id"]

nodes = pq.read_table(DATASET / "variants" / "nodes")
nodes = nodes.filter(pc.equal(nodes["variant_id"], variant)).sort_by("node_idx")
results = pq.read_table(DATASET / "scenarios" / "node_results")
results = results.filter(pc.equal(results["scenario_id"], sid)).sort_by("node_idx")

v = results["v_re"].to_numpy() + 1j * results["v_im"].to_numpy()  # V, to ground
vm_pu = np.abs(v) / (nodes["kv_base_ln"].to_numpy() * 1e3)  # per unit of the node's base
for k, (bus, conductor) in enumerate(zip(nodes["bus"].to_pylist(), nodes["conductor"].to_pylist())):
    if bus == "671":  # the three phases of one bus
        print(f"{bus}.{conductor}: {vm_pu[k]:.3f} pu at {np.angle(v[k], deg=True):7.1f} deg")
```

Output:

```text
671.a: 0.999 pu at    -3.5 deg
671.b: 1.036 pu at  -122.1 deg
671.c: 1.016 pu at   118.2 deg
```

### The network admittance and Kirchhoff's current law

The stored voltages solve the stored network, and one sparse product shows it. `Y_state` (S, rows
and columns are `node_idx`) is the network admittance of the scenario's state, assembled from the
Yprims of its power-delivery elements ([Outputs](outputs.md#statesybusparquet)). It is the graph a
model learns on, and `I_inj = Y_state V`, the current each node injects into the network, is a
physics check or a loss you can reuse on predicted voltages. Where no load, generator or source
connects, Kirchhoff's current law makes `I_inj` zero, up to the power flow's tolerance.
`DatasetReader` returns `Y_state` with the nodes and elements of the variant:

```python
from gridfm_datakit_distr.io.reader import DatasetReader

reader = DatasetReader(DATASET)
(scenario,) = reader.scenarios([sid])  # the scenario above, with its state's network
model = scenario.variant.model  # nodes and elements of its variant
Y = scenario.y_state  # network admittance of its state (S, sparse)
if Y is None:  # a scenario without a state: the variant's, as compiled
    Y = scenario.variant.y_base
i_inj = Y @ v  # A injected at each node

# the nodes where an element that is not power delivery (load, generator, source) connects
injecting = np.zeros(model.n_nodes, dtype=bool)
for e in model.elements:
    if not e.is_pd:
        injecting[[k for k in e.local_nodes() if k >= 0]] = True  # k < 0: ground
i_abs = np.abs(i_inj)
kcl = i_abs[~injecting].max() / i_abs.max()
assert kcl < 1e-6  # Kirchhoff: no net current where nothing is injected
print(
    f"KCL at the {(~injecting).sum()} of {model.n_nodes} nodes without injection: "
    f"|I_inj| below 1e-6 of the largest, {i_abs.max():.0f} A"
)
```

Output:

```text
KCL at the 19 of 41 nodes without injection: |I_inj| below 1e-6 of the largest, 563 A
```

### Power flows from the element currents

Line currents, power flows and losses are not stored by default (`labels.element_currents: true`
stores the currents), but they follow exactly from the voltages and each element's Yprim: the
current into its terminal conductors is `I = Yprim · V` and the complex power into a terminal is
`S = Σ V I*` over its conductors. They are the labels for branch quantities. The variant's model
carries the Yprims of the elements the network state does not change, which includes the lines
(regulated transformers and switched capacitors have theirs per state).

```python
v_ground = np.append(v, 0)  # index -1, the ground, then reads 0 V
flows = {}  # line name -> (power into terminal 1, loss), VA
for e in model.elements:
    if e.element_class == "line" and e.yprim is not None:
        v_e = v_ground[e.local_nodes()]  # V at each terminal conductor
        s = v_e * np.conj(e.yprim @ v_e)  # VA into the line at each terminal conductor
        flows[e.name] = (s[: e.n_conductors].sum(), s.sum())  # loss: in at both terminals

print("line                 kVA in   loss kW")
for name, (s_in, loss) in sorted(flows.items(), key=lambda item: -abs(item[1][0]))[:5]:
    print(f"{name:<18} {abs(s_in) / 1e3:8.0f} {loss.real / 1e3:9.2f}")
assert all(loss.real > -1e-9 * max(abs(s), 1) for s, loss in flows.values())  # lines dissipate
```

Output:

```text
line                 kVA in   loss kW
Line.650632            2038     16.58
Line.632670            1490      3.32
Line.670671            1384      6.44
Line.692675             638      1.79
Line.632633             335      0.40
```

### The power flow without OpenDSS

The tables alone determine the solution: the network admittance, the source (`variants/source`:
ideal voltages behind an admittance), the setpoints of the loads and PV units and the way each
divides its power between its branches (equal shares: [Outputs](outputs.md#scenariosloads)). The
reference loader reads them into arrays and a fixed-point power flow in NumPy solves them, without
importing OpenDSS. Solving them and comparing with the stored voltages closes the loop: the dataset
reproduces its own labels, with no engine in between.

```python
from gridfm_datakit_distr.validate.closed_loop import closed_loop

loop = closed_loop(scenario)  # solves the power flow from the tables, compares it with the stored v
assert loop.max_dv_pu < 1e-6  # pu of each node's base; float32 setpoints (float_precision: mixed)
print(f"the tables alone reproduce the stored voltages at all {len(v)} nodes within 1e-6 pu")
```

Output:

```text
the tables alone reproduce the stored voltages at all 41 nodes within 1e-6 pu
```

### The positive-sequence equivalent for gridfm-graphkit

`export-posseq` builds the positive-sequence equivalent of every scenario from the stored
solutions (the three-phase backbone and the demand each of its buses carries, re-solved in
OpenDSS) and writes it in gridfm-datakit's format, units and layout (MW, MVAr, degrees, per unit on
100 MVA; `bus_data.parquet`, `branch_data.parquet`, `gen_data.parquet`, `y_bus_data.parquet`,
`n_scenarios.txt`), one directory per network: point graphkit's `data_path` at
`<out>/<network_id>/raw`. `posseq_scenarios.parquet`, next to `raw/`, records the status of every
scenario's view ([Outputs §7.2](outputs.md#72-positive-sequence-export-export-posseq-not-stored-in-the-dataset),
[`export-posseq`](cli.md#export-posseq)).

```bash
uv run gridfm_datakit_distr export-posseq data/ieee13_small --out data/ieee13_small_datakit
```

Extra columns start with `x_`; `x_truth_Vm`, `x_truth_Va` are the positive-sequence voltages of
the full three-phase model at the same buses, so the error of the balanced approximation is
measurable:

```python
raw = Path("data/ieee13_small_datakit/ieee13/raw")
bus = pq.read_table(raw / "bus_data.parquet")
gap = np.abs(bus["Vm"].to_numpy() - bus["x_truth_Vm"].to_numpy()).max()
print(
    f"{bus.num_rows} bus rows, {int((raw / 'n_scenarios.txt').read_text())} scenarios; "
    f"largest |Vm - truth| {gap:.2e} pu"
)
assert gap < 0.01
```

Output:

```text
440 bus rows, 40 scenarios; largest |Vm - truth| 2.20e-03 pu
```

## 5. OPF labels

With the `opf` extra installed, a dataset can also carry the optimal power flow of every scenario:
the dispatch of the PV units that imports the least power within the voltage limits, checked in
OpenDSS ([OPF labels](opf.md)). `scripts/config/opf_small.yaml` asks for 10 scenarios of a small
CSIRO feeder with large PV units:

```bash
uv sync --extra opf
uv run --extra opf gridfm_datakit_distr generate scripts/config/opf_small.yaml --data-dir data
```

Every scenario has a row in `opf/opf_meta`, with its status and class; the solved ones have the
dispatch in `opf/opf_der` and the voltages in `opf/opf_node_results`:

```python
from collections import Counter

import pyarrow.parquet as pq

opf = pq.read_table("data/opf_small/opf/opf_meta").to_pylist()
print(Counter((r["opf_status"], r["opf_class"]) for r in opf))
solved = [r for r in opf if r["opf_status"] == "Solve_Succeeded"]
assert all(r["verified"] for r in solved)  # OpenDSS reproduces every dispatch
curtailed = {r["scenario_id"]: r["curtailment_w"] for r in solved if r["curtailment_w"] > 1.0}
print("curtailed (W):", curtailed)
dispatch = pq.read_table("data/opf_small/opf/opf_der")
pv = pq.read_table("data/opf_small/scenarios/der")
print(dispatch.num_rows, "PV set points of the OPF next to", pv.num_rows, "of the power flow")
```

In a class-`B` scenario the power flow's dispatch breaks a voltage limit and the OPF corrects it,
with reactive power first and by curtailing when that is not enough. The larger slice of five
feeders is `scripts/config/opf_slice.yaml` (500 scenarios, about 22 minutes on one process).

## 6. Scaling up

### Your own config

Copy a sample config from `scripts/config/` and change what you need; keys you leave out take
their defaults. Random quantities take *value specs*: a number, or a distribution such as
`{distribution: uniform, low: 0.9, high: 1.1}`. Every key, its default and the value specs are in
the [Configuration reference](config.md); `scripts/config/default.yaml` lists the main keys at
their defaults, with comments. A config is checked as a whole before anything runs: every problem
is listed as `<dotted.path>: <reason>`.

| Config | Network | What it shows | Size |
|---|---|---|---|
| `ieee123.yaml` | IEEE 123 | regulators and capacitors settled per scenario (`controls`), single-phase laterals | 1000 scenarios, ~5 MB |
| `csiro_feeder.yaml` | ENWL feeder n01_f1 (CSIRO) | four-wire LV with explicit neutral, phase reallocation, PV up to 60 % of the customers | 1000 scenarios, ~50 MB |
| `edp_rural.yaml` | EDP rural MV/LV | EDP's smart-meter profiles as loads instead of synthetic draws | 2000 scenarios, ~0.5 GB |
| `opf_slice.yaml` | five CSIRO feeders | OPF labels (section 5) | 500 scenarios |

### Work units and processes

A **work unit** is a chunk of at most `settings.chunk_size` (default 500) scenarios of one
variant: a dataset has networks × variants × ⌈scenarios per variant / `chunk_size`⌉ units (the dry
run prints the count). Units run in parallel, one per worker process: `--processes N`
(`settings.num_processes`, default one per CPU). With fewer units than processes, the extra
processes stay idle; a process that takes a unit of a variant it does not hold first compiles and
checks that variant. `--contexts N` runs up to N units of the same variant at once in each
process, one thread and OpenDSS context each.

```bash
uv run gridfm_datakit_distr generate scripts/config/csiro_feeder.yaml --data-dir data --processes 4
```

The data do not depend on the number of processes or contexts, nor on the order of the units
([reproducibility](scenarios.md#seeds-and-reproducibility)). Memory is per process:
`manifest.json` records, for every unit, the peak resident memory of the process that ran it
(`peak_rss_mib`); size `--processes` on a short run.

**Estimates.** The dry run's time and storage come from measured constants and are orders of
magnitude. Time grows with the customers (each one's setpoints are set and checked through the
engine); networks with regulator or capacitor controls take longer, up to 22 times the estimate on
IEEE 8500, and the dry run says so. OPF solves and the Kron view's solves are not in the time
estimate. For scale: 10⁶ snapshots of CSIRO LV feeders take about 0.3 h on 32 cores and 60 GB
(extrapolated from 10⁵).

### Long runs

- Run `generate` detached (`tmux`, `nohup`, a batch job). Ctrl-C, a killed worker or a crash stops
  it; every file is written to a temporary name and renamed, and `manifest.json` records each
  finished unit, so the same command resumes where it stopped.
- `--processes` and `--contexts` may differ between the runs of a dataset, and a moved dataset
  resumes from its new `--data-dir`; any other change makes another dataset
  ([rules](cli.md#generate)). To grow a dataset, generate a new one: with the same seed it repeats
  the smaller dataset's draws and solutions ([Scenarios](scenarios.md#seeds-and-reproducibility)).
- One `generate` per dataset directory at a time.

## 7. Troubleshooting

| Message or symptom | Cause | What to do |
|---|---|---|
| `error: run gridfm_datakit_distr fetch <source> first` | the source is not in this cache | `fetch` it, or point `--cache` / `$GRIDFM_DISTR_CACHE` at the cache that has it |
| `error: invalid configuration …` (exit 2) | the listed keys | fix each line; unknown keys are errors, `"1"` is not a number, and YAML reads `1e-10` as text (write `1.0e-10`) |
| `labels.opf.enabled: OPF labels need the 'opf' extra` | CasADi not installed | `uv sync --extra opf`, then `uv run --extra opf …` |
| `… holds a dataset of another config or version` | config, generator or OpenDSS changed since the dataset was started | another `dataset_name`, or `--overwrite` (deletes the old dataset) |
| `… exists and is not a dataset (no manifest.json): not touched` | the output directory is not empty | another `dataset_name` or `--data-dir` |
| `error: N unit(s) failed, e.g. …` | see each unit's `message` in `manifest.json` | a rerun retries failed units; a failure caused by the drawn data repeats |
| … `variant failed: attempt 0: C1: …` | the variant's base case did not converge or failed the Kirchhoff check, in all three attempts ([Variants](variants.md#checks-and-failures)) | smaller perturbations; CSIRO `n13_f4` fails at its nominal loads ([Networks](networks.md#known-differences-and-exclusions)) |
| `scenarios.profiles.provider: <network_id>: no load has a daily or yearly loadshape` | `network_loadshapes` or `edp_smart_meter` on a network without loadshapes | use them on the EDP networks and `ieee_eulv` only ([Scenarios](scenarios.md#loads)) |
| many `not_converged` or `off_setpoint` scenarios | operating points the network cannot carry at constant power (they are recorded, not errors) | `stats` and the dataset card count them, `pf_message` explains them; if unwanted, narrow the load, PV or source-voltage ranges |
| the run stops with a dead worker process | a worker was killed, e.g. by the out-of-memory killer | resume with fewer `--processes` |
| `export-posseq`: `incomplete in the dataset` | a network with failed or unfinished units | resume `generate`, or `--network` for the complete ones |
| `--plots needs matplotlib` | `plots` extra missing | install it ([Installation](../installation.md#optional-extras)) |
| `croissant`: `validation needs mlcroissant` | `croissant` extra missing | install it ([Installation](../installation.md#optional-extras)), or `--no-validate` |

## 8. Where next

- [Networks](networks.md): the catalog, sources, representations and the conformance suite.
- [Variants](variants.md) and [Scenarios](scenarios.md): what changes from one sample to the next,
  statuses, seeds and reproducibility.
- [OPF labels](opf.md): the OPF problem, its scope and its statuses.
- [Outputs](outputs.md): every table and column, units and conventions.
- [Configuration reference](config.md): every config key with its default.
- [Command-line reference](cli.md): every command and option.
