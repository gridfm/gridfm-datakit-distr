# Getting started

This guide takes you from an empty machine to a dataset of three-phase unbalanced power flows
and shows how to read it. Every command and every Python snippet below is run by the test suite
(`integrationtests/test_tutorial.py`), so they work as written.

**What a dataset is.** A distribution network from the catalog (an IEEE test feeder, a
four-wire LV feeder of the ENWL networks, an EDP MV/LV network) is perturbed into *variants*
(line lengths, earthing resistances, phases of customers, rooftop PV). Each variant is solved for
many *scenarios* (loads, PV output, source voltage) in OpenDSS, the reference engine. Every node
of every converged scenario gets its complex voltage. The admittances of the network and of each
element are stored too, so every current and power flow follows from the voltages (section 4
below). The dataset is this three-phase truth. Simplified versions are derived from it on demand:
`export-posseq` writes the positive-sequence equivalent in gridfm-datakit's layout for today's
GENCO, and the Kron view (neutral eliminated) is re-solved next to the full model when the config
asks for it. Tables and units are defined in [Outputs](outputs.md), config keys
in [Configuration reference](config.md).

## 1. Install and fetch the networks

You need Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repository> gridfm-datakit-distr && cd gridfm-datakit-distr
uv sync
uv run gridfm_datakit_distr fetch ieee13 ieee123 csiro_enwl   # downloads with checksums
uv run gridfm_datakit_distr inspect ieee13                    # optional: the network's summary
```

Networks are cached in `~/.cache/gridfm_datakit_distr` (or `$GRIDFM_DISTR_CACHE`); nothing of them
goes into the repository. EDP's archives are RAR files: `fetch edp` needs `unrar`, `7z` or `bsdtar`.

## 2. A first dataset: IEEE 13 in a few seconds

`scripts/config/ieee13_small.yaml` asks for 2 variants × 20 scenarios of IEEE 13. Look before you run:

```bash
uv run gridfm_datakit_distr generate scripts/config/ieee13_small.yaml --dry-run
```

The dry run loads the network and prints the number of scenarios and work units, and estimates of
time and storage. Then generate and summarise:

```bash
uv run gridfm_datakit_distr generate scripts/config/ieee13_small.yaml --data-dir data
uv run gridfm_datakit_distr stats data/ieee13_small
```

The dataset is in `data/ieee13_small`. Running the same command again *resumes*: finished units
are kept, missing ones are computed (an interrupted run loses nothing). A different config with
the same `dataset_name` is refused unless you pass `--overwrite`. `--processes N` sets the number
of worker processes; on one machine the dataset is the same bit for bit whatever `N`
([reproducibility](opf.md#reproducibility)).

## 3. Three larger examples

| Config | Network | What it shows | Size |
|---|---|---|---|
| `scripts/config/ieee123.yaml` | IEEE 123 | regulators and capacitors settled per scenario (`controls`), single-phase laterals | 1000 scenarios, ~5 MB |
| `scripts/config/csiro_feeder.yaml` | ENWL feeder n01_f1 (CSIRO) | four-wire LV with explicit neutral, earthing resistances, phase reallocation, PV up to 60 % of the customers | 1000 scenarios, ~50 MB |
| `scripts/config/edp_rural.yaml` | EDP rural MV/LV | the loads of EDP's smart-meter profiles instead of synthetic draws | 2000 scenarios, ~0.5 GB |

```bash
uv run gridfm_datakit_distr generate scripts/config/ieee123.yaml --dry-run
uv run gridfm_datakit_distr generate scripts/config/csiro_feeder.yaml --data-dir data --processes 4
```

To make your own, copy one of them and change what you need; every key and its default is in
[Configuration reference](config.md). Value specs take a number or a distribution:
`{distribution: uniform, low: 0.9, high: 1.1}`, `{distribution: normal, mean: 1, std: 0.05}`,
`{distribution: choice, values: [1, 5, 20]}`.

## 4. Reading the data

The tables are Parquet files: a directory of parts (`scenarios/node_results/part-*.parquet`) or a
single file (`states/ybus.parquet`); `pyarrow.parquet.read_table` reads both. The snippets use the
IEEE 13 dataset of section 2.

```python
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

DATASET = Path("data/ieee13_small")
meta = pq.read_table(DATASET / "scenarios" / "scenario_meta")
print(meta.num_rows, "scenarios:", pc.value_counts(meta["pf_status"]).to_pylist())
```

Each scenario has a `scenario_id`, its `variant_id`, the `state_id` of its network state (tap
positions and capacitor steps) and a `pf_status`; only `converged` scenarios have results.

### Voltages of a scenario

Nodes are the conductors of the buses (`a`, `b`, `c`, `n`, …) of a variant, numbered by
`node_idx`; voltages are to ground, in volts, as two float columns.

```python
first = meta.filter(pc.equal(meta["pf_status"], "converged")).slice(0, 1).to_pylist()[0]
sid, variant, state = first["scenario_id"], first["variant_id"], first["state_id"]

nodes = pq.read_table(DATASET / "variants" / "nodes")
nodes = nodes.filter(pc.equal(nodes["variant_id"], variant)).sort_by("node_idx")
results = pq.read_table(DATASET / "scenarios" / "node_results")
results = results.filter(pc.equal(results["scenario_id"], sid)).sort_by("node_idx")

v = results["v_re"].to_numpy() + 1j * results["v_im"].to_numpy()  # V, to ground
vm_pu = np.abs(v) / (nodes["kv_base_ln"].to_numpy() * 1e3)
for bus, conductor, vm in list(
    zip(nodes["bus"].to_pylist(), nodes["conductor"].to_pylist(), vm_pu)
)[:5]:
    print(f"{bus}.{conductor}: {vm:.4f} pu")
```

### The network admittance and Kirchhoff's current law

`states/ybus` holds the network admittance `Y_state` of every state (siemens, `row`/`col` are
`node_idx`). It maps the voltages to the currents that loads, generators and sources inject into
the network, `I_inj = Y_state V` (not stored: one sparse product). At a node where only lines,
transformers and other power-delivery elements meet, Kirchhoff's current law makes `I_inj` zero:
a check of the stored solution, up to the power flow's tolerance.

```python
def table(path, column, value):
    t = pq.read_table(path)
    return t.filter(pc.equal(t[column], value))


ybus = table(DATASET / "states" / "ybus.parquet", "state_id", state)
n = nodes.num_rows
Y = sp.coo_matrix(
    (
        ybus["g"].to_numpy() + 1j * ybus["b"].to_numpy(),
        (ybus["row"].to_numpy(), ybus["col"].to_numpy()),
    ),
    shape=(n, n),
).tocsr()
i_inj = Y @ v  # A

elements = table(DATASET / "variants" / "elements", "variant_id", variant)
terminals = table(DATASET / "variants" / "element_terminals", "variant_id", variant)
conversion = set(elements.filter(pc.invert(elements["is_pd"]))["element_idx"].to_pylist())
connected = {
    k
    for e, k in zip(terminals["element_idx"].to_pylist(), terminals["node_idx"].to_pylist())
    if e in conversion
}
free = [k for k in range(n) if k not in connected]
kcl = np.abs(i_inj[free]).max() / np.abs(i_inj).max()
print(
    f"KCL at the {len(free)} of {n} nodes without loads or sources: {kcl:.1e} of the largest |I_inj|"
)
assert kcl < 1e-6
```

`Y_state` is assembled from the primitive admittances (Yprim) of the power-delivery elements:
`variants/yprim` for the elements whose Yprim does not depend on the network state, and
`states/yprim_state` for the others (regulated transformers, switched capacitors). A Yprim's
`row`/`col` are local positions, terminal by terminal (`terminal · n_conductors + position`), and
`variants/element_terminals` gives the node of each position (`-1`: ground, left out):

```python
yprims = [table(DATASET / "variants" / "yprim", "variant_id", variant)]
if (DATASET / "states" / "yprim_state.parquet").exists():
    yprims.append(table(DATASET / "states" / "yprim_state.parquet", "state_id", state))

n_conductors = dict(zip(elements["element_idx"].to_pylist(), elements["n_conductors"].to_pylist()))
node_of = {
    (e, t * n_conductors[e] + p): k
    for e, t, p, k in zip(
        *(terminals[c].to_pylist() for c in ("element_idx", "terminal", "position", "node_idx"))
    )
}
rows, cols, values = [], [], []
for y in yprims:
    for e, r, c, g, b in zip(*(y[k].to_pylist() for k in ("element_idx", "row", "col", "g", "b"))):
        kr, kc = node_of[(e, r)], node_of[(e, c)]
        if kr >= 0 and kc >= 0:
            rows.append(kr)
            cols.append(kc)
            values.append(complex(g, b))
Y_assembled = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
print("assembled vs stored Y_state:", abs(Y_assembled - Y).max() / abs(Y).max())
assert abs(Y_assembled - Y).max() <= 1e-12 * abs(Y).max()
```

### Power flows from the element currents

The current into an element at each of its terminal conductors is its Yprim times the voltages
of its nodes, `I = Yprim · V` (exactly what the generator computes; with
`labels.element_currents: true` the dataset also stores them, in `scenarios/element_results`).
With the voltages they give the complex power into the element at each terminal, `S = Σ V I*`;
the sum over its terminals is the element's loss.

```python
size = {
    e: n_t * n_c
    for e, n_t, n_c in zip(
        *(elements[c].to_pylist() for c in ("element_idx", "n_terminals", "n_conductors"))
    )
}
yprim_of = {}
for y in yprims:
    for e, r, c, g, b in zip(*(y[k].to_pylist() for k in ("element_idx", "row", "col", "g", "b"))):
        yprim_of.setdefault(e, np.zeros((size[e], size[e]), dtype=complex))[r, c] = complex(g, b)

names = dict(zip(elements["element_idx"].to_pylist(), elements["name"].to_pylist()))
power = {}
for e, y_e in yprim_of.items():
    nodes_e = [node_of[(e, q)] for q in range(size[e])]
    v_local = np.array([v[k] if k >= 0 else 0.0 for k in nodes_e])
    i_local = y_e @ v_local  # A, into the element
    n_c = n_conductors[e]
    for t in range(size[e] // n_c):
        part = slice(t * n_c, (t + 1) * n_c)
        power[(e, t)] = np.sum(v_local[part] * np.conj(i_local[part]))

lines = [e for e, name in names.items() if name.startswith("Line.") and (e, 1) in power]
line = max(lines, key=lambda e: abs(power[(e, 0)]))  # the most loaded line
s_from, s_to = power[(line, 0)], power[(line, 1)]
print(
    f"{names[line]}: {s_from / 1e3:.1f} kVA in at terminal 1, losses {(s_from + s_to) / 1e3:.3f} kVA"
)
assert (s_from + s_to).real > -1e-9 * abs(s_from)  # a passive line dissipates
```

### The power flow without OpenDSS

The tables determine the voltages: the network admittance, the source (`variants/source`: ideal
voltages behind an admittance), the setpoints of the loads and DER and the way each divides its
power between its branches (equal shares, [Outputs](outputs.md), `scenarios/loads` in §6). The reference loader reads them into arrays
and a fixed-point power flow in NumPy solves them; neither imports OpenDSS. It is also the place of
two baselines a learned model has to beat: the no-load voltage and one linearised step from it.

```python
from gridfm_datakit_distr.io.reader import DatasetReader
from gridfm_datakit_distr.validate.closed_loop import closed_loop, linearised, no_load

reader = DatasetReader(DATASET)
(scenario,) = reader.scenarios([sid])
base = scenario.variant.base_v
loop = closed_loop(scenario)
print(f"fixed point in {loop.iterations} iterations: {loop.max_dv_pu:.1e} pu from the stored voltages")
for name, baseline in (("no load", no_load), ("linearised", linearised)):
    print(f"{name}: {(np.abs(baseline(scenario) - v) / base).max():.1e} pu at most")
assert loop.max_dv_pu < 1e-6  # float32 setpoints (settings.float_precision: mixed)
```

### The positive-sequence equivalent for gridfm-graphkit

`export-posseq` builds the positive-sequence equivalent of every scenario from the stored
solutions: the three-phase backbone, the demand each of its buses carries, re-solved in OpenDSS
([Outputs](outputs.md), §7.2). It writes it in gridfm-datakit's format, units and layout (MW, MVAr, degrees, per
unit on 100 MVA; `bus_data.parquet`, `branch_data.parquet`, `gen_data.parquet`,
`y_bus_data.parquet`, `n_scenarios.txt`), one directory per network: point graphkit's
`data_path` at `<out>/<network_id>/raw`. `--processes N` builds variants in parallel;
`posseq_scenarios.parquet`, next to `raw/`, records the status of every scenario's view.

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
    f"largest |Vm - truth| {gap:.1e} pu"
)
assert gap < 0.01
```

### Statistics and metadata

`stats` writes `stats/stats.json` (distributions of phase and neutral voltages, voltage unbalance
factor, loadings, violation rates; `--plots` also draws them, with the `plots` extra);
`croissant.json` describes every table in MLCommons Croissant (see [Installation](../installation.md));
`README.md` is the dataset card; `manifest.json` records the resolved config, its hash and the
versions that made the data.

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

## 6. Where next

- [Networks](networks.md): the catalog, where each network comes from and its conformance status.
- [Variants](variants.md) and [Scenarios](scenarios.md): what changes from one sample to the next.
- [OPF labels](opf.md): the OPF problem, its scope and its statuses.
- [Outputs](outputs.md): every table and column, units and conventions.
- [Configuration reference](config.md): every config key with its default.
