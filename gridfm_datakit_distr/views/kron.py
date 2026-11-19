"""Kron view of a network: the neutral eliminated, re-solved in OpenDSS.

The view is the variant's network plus OpenDSS edits:

- energy meters are disabled (measurement only; with a disabled element in their zone the solution
  is wrong);
- a line (or switch) with neutral conductors is disabled and replaced by ``<name>_kron`` on its
  phase nodes, with the series matrix reduced by the Schur complement ``Z_pp - Z_pn Z_nn^-1 Z_np``
  and the phase block of its capacitance (written whole with its unit and length);
- loads, generators, PV, storage, capacitors and reactors drop their neutral nodes from ``bus1`` (a
  wye element then returns through ground);
- grounding reactors (neutral conductors to ground) are disabled;
- a wye transformer winding with a neutral node gets its star point on node 0, solidly
  (``Rneut=-1``).

The neutral is thus at 0 V everywhere: the assumption of Kron reduction. A network without neutral
conductor nodes (Kron-native) has no view to build: :func:`kron_script` returns None.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_semantic import is_grounding_reactor
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.state import NetworkState
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.scenarios.controls import apply_state, model_at_state
from gridfm_datakit_distr.scenarios.profiles import customers_of
from gridfm_datakit_distr.solve.pf_batch import (
    ScenarioOutcome,
    VariantRunner,
    engine_indices,
    set_setpoints,
)
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from gridfm_datakit_distr.variants.generate import VariantResult, physics_failures

_SHUNT_OR_PC = {"load", "generator", "pvsystem", "storage", "capacitor", "reactor"}


class KronError(RuntimeError):
    """An element connected to a neutral that the Kron view does not handle."""


def kron_script(ctx: DssContext, model: ElectricalModel) -> list[str] | None:
    """Edits that turn a compiled network (``ctx``, its ``model``) into its Kron view; None if it
    has no neutral node."""
    if not any(n.conductor == "n" for n in model.nodes):
        return None
    # an EnergyMeter whose element gets disabled breaks OpenDSS's re-processing of the buses: nodes
    # of disabled elements stay and the solve converges to a wrong solution
    meters = [m for m in ctx.api.Meters.AllNames() if m and m.lower() != "none"]
    script: list[str] = [f"Disable EnergyMeter.{m}" for m in meters]
    for e in sorted(model.elements, key=lambda e: e.name):
        if not e.enabled or not _touches_neutral(e, model):
            continue
        cls = e.element_class
        if cls in ("line", "switch"):
            script += _line(ctx, model, e)
        elif cls == "reactor" and is_grounding_reactor(e, model):
            script.append(f"Disable {e.name}")
        elif cls in _SHUNT_OR_PC:
            script.append(f"Edit {e.name} bus1={_bus_without_neutral(e.terminals[0], model)}")
        elif cls == "transformer":
            script += _transformer(e, model)
        else:
            raise KronError(f"{e.name}: class {cls!r} connected to a neutral is not handled")
    return script


def _touches_neutral(e: Element, model: ElectricalModel) -> bool:
    return any(k != GROUND and model.nodes[k].conductor == "n" for t in e.terminals for k in t)


def _suffix(k: int, model: ElectricalModel) -> str:
    return "0" if k == GROUND else model.nodes[k].opendss_node.rsplit(".", 1)[1]


def _bus_of(terminal: list[int], model: ElectricalModel) -> str:
    bus = next(model.nodes[k].bus for k in terminal if k != GROUND)
    return bus


def _bus_without_neutral(terminal: list[int], model: ElectricalModel) -> str:
    kept = [k for k in terminal if k == GROUND or model.nodes[k].conductor != "n"]
    nodes = [_suffix(k, model) for k in kept if k != GROUND]
    return ".".join([_bus_of(terminal, model), *nodes])


def _line(ctx: DssContext, model: ElectricalModel, e: Element) -> list[str]:
    api = ctx.api
    name = e.name.split(".", 1)[1]
    api.Lines.Name(name)
    n = len(e.terminals[0])
    r = np.asarray(api.Lines.RMatrix(), dtype=np.float64).reshape(n, n)
    x = np.asarray(api.Lines.XMatrix(), dtype=np.float64).reshape(n, n)
    c = np.asarray(api.Lines.CMatrix(), dtype=np.float64).reshape(n, n)
    length = float(api.Lines.Length())
    units = ctx.command(f"? Line.{name}.units").strip()
    neutral = [
        p
        for p in range(n)
        if any(t[p] != GROUND and model.nodes[t[p]].conductor == "n" for t in e.terminals)
    ]
    phases = [p for p in range(n) if p not in neutral]
    z = r + 1j * x
    z_red = z[np.ix_(phases, phases)] - z[np.ix_(phases, neutral)] @ np.linalg.solve(
        z[np.ix_(neutral, neutral)], z[np.ix_(neutral, phases)]
    )
    c_red = c[np.ix_(phases, phases)]
    buses = [
        ".".join([_bus_of(t, model), *(_suffix(t[p], model) for p in phases)]) for t in e.terminals
    ]
    return [
        f"Disable {e.name}",
        f"New Line.{name}_kron bus1={buses[0]} bus2={buses[1]} phases={len(phases)} "
        f"rmatrix={_lower(z_red.real)} xmatrix={_lower(z_red.imag)} cmatrix={_lower(c_red)} "
        f"units={units} length={length!r}",
    ]


def _transformer(e: Element, model: ElectricalModel) -> list[str]:
    out = []
    for w, terminal in enumerate(e.terminals, start=1):
        neutral = [k != GROUND and model.nodes[k].conductor == "n" for k in terminal]
        if any(neutral):
            # ground the neutral in place: conductor order is polarity (centre-tap ``x.3.2``)
            nodes = [
                "0" if n else _suffix(k, model) for k, n in zip(terminal, neutral, strict=True)
            ]
            bus = ".".join([_bus_of(terminal, model), *nodes])
            out.append(f"Edit {e.name} wdg={w} bus={bus} Rneut=-1")
    return out


def _lower(m: np.ndarray) -> str:
    rows = [" ".join(repr(float(m[i, j])) for j in range(i + 1)) for i in range(m.shape[0])]
    return "[" + " | ".join(rows) + "]"


# ---------------------------------------------------------------------------- the view per scenario


@dataclass(frozen=True, slots=True, eq=False)
class KronOutcome:
    status: str
    """``converged``, ``off_setpoint``, ``not_converged`` or ``error``."""
    v: NDArray[np.complex128] | None
    """Node voltages of the view (V, view node order) when converged at the setpoints."""
    message: str | None = None


class KronView:
    """The Kron view of a variant, solved with the full model's setpoints and network state."""

    def __init__(
        self,
        runner: VariantRunner,
        network: Network,
        variant: VariantResult,
        *,
        root: Path | None = None,
    ) -> None:
        self.runner = runner
        self.variant_id = variant.variant_id
        self._met: set[str] = set()
        self._new: dict[str, ElectricalModel] = {}
        full = runner.model
        script = kron_script(runner.ctx, full)
        self.identity = script is None
        if script is None:
            kept = [n for n in full.nodes if n.conductor != "n"]
            self.nodes = [(n.bus, n.conductor, n.idx) for n in kept]
            return
        prepared = prepare_network(
            network, runner.options, root, variant_id=variant.variant_id,
            edits=[*variant.script, *script],
        )  # fmt: skip
        failures = physics_failures(prepared)
        if failures:
            raise KronError(f"{variant.variant_id}: the Kron view fails its base case: {failures}")
        self.ctx, self.model = prepared.ctx, prepared.model
        if [c.element for c in customers_of(prepared)] != [c.element for c in runner.customers]:
            raise KronError(f"{variant.variant_id}: the Kron view has other loads than the model")
        by_name = {n.opendss_node: n.idx for n in full.nodes}
        self.nodes = [(n.bus, n.conductor, by_name[n.opendss_node]) for n in self.model.nodes]
        self._to_model = self.ctx.ynode_positions(n.opendss_node for n in self.model.nodes)
        api = self.ctx.api
        self._load_idx = engine_indices(api.Loads.AllNames(), (c.element for c in runner.customers))
        units = runner.pv.units if runner.pv is not None else ()
        self._pv_idx = engine_indices(api.Generators.AllNames(), (u.generator for u in units))
        self.ctx.api.Vsources.First()
        self._source = str(self.ctx.api.Vsources.Name())

    def solve(self, outcome: ScenarioOutcome, state: NetworkState | None) -> KronOutcome:
        """The view of a scenario whose full model converged: same setpoints, same state."""
        if self.identity:
            return KronOutcome("converged", outcome.v)
        try:
            set_setpoints(
                self.ctx.api, self._load_idx, outcome.load_p_w, outcome.load_q_var, self._pv_idx,
                outcome.pv_p_w, outcome.pv_q_var, self._source, outcome.source_v_pu,
            )  # fmt: skip
            self.ctx.command("Set ControlMode=OFF")
            if state is not None:
                apply_state(self.ctx, state)
            self.ctx.command("Set Mode=Snap")
            status = self.ctx.solve()
            if not status.converged:
                return KronOutcome("not_converged", None)
            if state is not None and state.state_id not in self._met:
                self._met.add(state.state_id)
                self._new[state.state_id] = model_at_state(self.model, self.ctx)
            if not setpoint_report(self.ctx).passes():
                return KronOutcome("off_setpoint", None)
            return KronOutcome("converged", self.ctx.ynode_voltages()[self._to_model])
        except Exception as exc:  # recorded, never dropped
            return KronOutcome("error", None, f"{type(exc).__name__}: {exc}")

    def begin_unit(self) -> None:
        """Start a work unit: :meth:`new_states` then reports every state the unit meets."""
        self._met = set()
        self._new = {}

    def new_states(self) -> dict[str, ElectricalModel]:
        """View models of the states met since the last call or :meth:`begin_unit` (for
        ``views/kron/ybus``)."""
        out, self._new = self._new, {}
        return out

    # ------------------------------------------------------------------------ tables

    def nodes_table(self) -> pa.Table:
        return pa.table(
            {
                "variant_id": [self.variant_id] * len(self.nodes),
                "node_idx_kron": np.arange(len(self.nodes), dtype=np.int32),
                "bus": [b for b, _, _ in self.nodes],
                "conductor": [c for _, c, _ in self.nodes],
                "node_idx_full": np.array([f for _, _, f in self.nodes], dtype=np.int32),
            },
            schema=schema.KRON_NODES,
        )

    def results_table(self, pairs: list[tuple[ScenarioOutcome, KronOutcome]]) -> pa.Table:
        """``views/kron/node_results`` of scenarios whose full model converged; empty for a
        Kron-native network (stored only as the mapping, outputs.md §7.1)."""
        if self.identity:
            return schema.KRON_NODE_RESULTS.empty_table()
        n = len(self.nodes)
        parts = []
        for o, k in pairs:
            if o.v is None:
                continue
            v = k.v if k.v is not None else np.full(n, np.nan + 0j)
            parts.append(
                pa.table(
                    {
                        "scenario_id": np.full(n, o.scenario_id, dtype=np.int64),
                        "node_idx_kron": np.arange(n, dtype=np.int32),
                        "v_re": pa.array(v.real, mask=np.isnan(v.real)),
                        "v_im": pa.array(v.imag, mask=np.isnan(v.imag)),
                        "view_status": [k.status] * n,
                    }
                ).cast(schema.KRON_NODE_RESULTS)
            )
        return pa.concat_tables(parts) if parts else schema.KRON_NODE_RESULTS.empty_table()
