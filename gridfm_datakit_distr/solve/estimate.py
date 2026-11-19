"""Dry-run estimates of a generation: counts, time and storage.

Rows per scenario are counted on each prepared network (nodes, extraction rows, customers, the
Kron view's nodes). Bytes per row and seconds per scenario are measurements (zstd parts written
by the executor, 6-core workstation):

- time per scenario ``≈ 3.7 ms + 0.03 µs · nodes + 22 µs · customers`` without views
  (fitted on CSIRO n21_f3 and EDP rural, which have no controls, with the per-node term of an
  earlier fit; EDP industrial 17 % below, CSIRO n01_f1 3 % above). The customers dominate: each
  one's setpoints are set and checked through the engine's API. Networks with regulator or
  capacitor controls settle every scenario: IEEE 13 takes 1.2 times the estimate, IEEE 8500 22
  times. The Kron view, when on, adds its own solve (not in the estimate);
- bytes per row as below (mixed and full precision; the views are float64 in both), plus a fixed
  part per variant (its model tables, admittance and semantic layer, ~500 bytes per node).

These are estimates of the order of magnitude, for planning; the run itself measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.config.value_spec import ValueSpec
from gridfm_datakit_distr.networks.prepare import PreparedNetwork
from gridfm_datakit_distr.scenarios.profiles import customers_of
from gridfm_datakit_distr.solve.extraction import build_plan

SECONDS = (3.7e-3, 0.03e-6, 22.4e-6)
"""Seconds per scenario: constant, per node, per customer (measured without views)."""
CONTROLS_FACTOR = 22.0
"""Measured on IEEE 8500 (12 regulators, 9 capacitor controls) against the estimate."""
BYTES = {
    "mixed": {"meta": 357.0, "node": 10.6, "element": 4.2, "load": 8.4, "der": 7.8},
    "full": {"meta": 373.0, "node": 10.6, "element": 6.3, "load": 16.3, "der": 12.6},
}
"""Bytes per row of the scenario tables in chunks of hundreds of scenarios: the part files of
CSIRO n21_f3, one variant of 2000 scenarios in chunks of 500, PV at 30 % of the customers, written
by generator 0.3.0.dev1 (``node``: voltages only, float64 in both precisions; ``element`` only with
``labels.element_currents``; ``meta`` is high because every row group of 16 scenarios carries its
own dictionaries)."""
VIEW_BYTES = {"kron": 11.3}
"""Bytes per Kron phase node and scenario, its state tables included (CSIRO n21_f3, 500
scenarios, 2532 phase nodes; the writer of ``BYTES``)."""
BYTES_PER_NODE_AND_VARIANT = 500.0
"""Model tables, admittance and semantic layer of a variant (CSIRO n21_f3: ``variants/``, 1.69 MB
for 3376 nodes)."""


@dataclass(frozen=True, slots=True)
class NetworkEstimate:
    network_id: str
    nodes: int
    customers: int
    kron_nodes: int
    controls: bool
    """Regulator or capacitor controls: the time estimate may be several times low."""
    seconds_per_scenario: float
    bytes_per_scenario: float
    bytes_per_variant: float


def mean_of(spec: ValueSpec, draws: int = 4000) -> float:
    """Mean of a value spec by sampling (fixed seed: the estimate is deterministic)."""
    rng = np.random.default_rng(0)
    return float(np.mean([float(spec.sample(rng)) for _ in range(draws)]))


def estimate_network(prepared: PreparedNetwork, config: GenerationConfig) -> NetworkEstimate:
    """Rows, seconds and bytes per scenario of one prepared network under ``config``."""
    model, api = prepared.model, prepared.ctx.api
    nodes = len(model.nodes)
    customers = len(customers_of(prepared))
    element_rows = int(build_plan(model).element_idx.size)
    neutral = any(n.conductor == "n" for n in model.nodes)
    kron_nodes = sum(n.conductor != "n" for n in model.nodes) if neutral else 0
    controls = bool(api.RegControls.Count() or api.CapControls.Count())
    pv = customers * mean_of(config.der.pv.penetration)
    b = BYTES[config.settings.float_precision]
    size = b["meta"] + b["node"] * nodes + b["load"] * customers + b["der"] * pv
    if config.labels.element_currents:
        size += b["element"] * element_rows
    if config.views.kron:
        size += VIEW_BYTES["kron"] * kron_nodes
    seconds = SECONDS[0] + SECONDS[1] * nodes + SECONDS[2] * customers
    return NetworkEstimate(
        prepared.network.id, nodes, customers, kron_nodes, controls, seconds,
        size, BYTES_PER_NODE_AND_VARIANT * nodes,
    )  # fmt: skip
