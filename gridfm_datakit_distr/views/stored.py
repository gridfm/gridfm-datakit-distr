"""The stored truth of a variant, read back for the views computed after generation.

The positive-sequence view is built from a dataset, not during generation, so everything it needs
comes from the stored tables:

- :class:`StoredVariant`: the variant's model (rebuilt with ``model.io_parquet.from_tables``) and
  its extraction plan, the customers and PV units in the order of ``scenarios/loads`` and
  ``scenarios/der``, the network's frequency and power base (``catalog/networks``, or the
  configured ``normalisation.s_base_va``), the solver tolerance of the generation, and the model
  at any network state of the variant;
- :class:`StoredScenario`: a converged scenario, with its voltages and the currents into the
  power-delivery elements recomputed as ``Yprim · V`` at its state, the extraction's own formula
 , so that they equal what the generation computed;
- :class:`VariantStore`: reads both for variant ``k`` of network ``i``, one chunk of scenarios at
  a time (memory stays bounded), in ``scenario_id`` order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io.reader import VariantTables
from gridfm_datakit_distr.io.writer import part_key, read_states
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.io_parquet import NODES, YPRIM_STATE
from gridfm_datakit_distr.solve.extraction import MAX_CACHED_STATES, ExtractionPlan, build_plan


@dataclass(frozen=True, slots=True)
class StoredVariant:
    """What the positive-sequence view needs of a variant (in place of the generation's runner)."""

    variant_id: str
    model: ElectricalModel
    """The variant's model at one of its network states (topology, ratings and Yprims)."""
    plan: ExtractionPlan
    customer_elements: tuple[str, ...]
    """Load names, in the order of each scenario's ``scenarios/loads`` rows."""
    pv_generators: tuple[str, ...]
    """PV generator names, in the order of each scenario's ``scenarios/der`` rows."""
    frequency_hz: float
    tolerance: float
    """OpenDSS ``Tolerance`` of the generation's solves."""
    s_base_va: float
    """Power base of the network (VA): the scale of the view's source limits and balance check."""
    state_model: Callable[[str], ElectricalModel]
    """The variant's model at a network state (tap-dependent Yprims of that state)."""


@dataclass(frozen=True, slots=True)
class StoredScenario:
    """A converged scenario of the full model, as stored (currents recomputed)."""

    scenario_id: int
    variant_id: str
    state_id: str
    v: NDArray[np.complex128]
    """Node voltages to ground (V), in the model's node order."""
    i_pd: NDArray[np.complex128]
    """Currents into the PD elements per local position (A): ``Yprim · V`` at the state."""
    load_p_w: NDArray[np.float64]
    load_q_var: NDArray[np.float64]
    pv_p_w: NDArray[np.float64] | None
    pv_q_var: NDArray[np.float64] | None
    pv_p_avail_w: NDArray[np.float64] | None


class VariantStore:
    """The stored truth of variant ``k`` of network ``i`` in the dataset at ``root``."""

    def __init__(self, root: Path, i: int, k: int, config: GenerationConfig) -> None:
        self.root, self.i, self.k = root, i, k
        key = part_key(i, k)
        variants = pq.read_table(root / "variants/variants" / f"part-{key}.parquet")
        self.variant_id = str(variants.column("variant_id")[0].as_py())
        self.stored = VariantTables(root, key, self.variant_id, read_states(root, YPRIM_STATE))
        self._models: dict[str, tuple[ElectricalModel, ExtractionPlan]] = {}
        """Per state: the model and plan; the most recent ``MAX_CACHED_STATES``."""
        networks = pq.read_table(root / "catalog" / "networks.parquet")
        row = networks.filter(pc.equal(networks["network_idx"], pa.scalar(i, pa.int32())))
        if row.num_rows != 1:
            raise ValueError(f"{root}: network {i} is not in catalog/networks")
        self.frequency_hz = float(row.column("frequency_hz")[0].as_py())
        explicit = config.normalisation.s_base_va
        stored = float(row.column("s_base_va")[0].as_py())
        self.s_base_va = stored if isinstance(explicit, str) else float(explicit)
        self.tolerance = config.settings.validation_tolerance
        self._meta_parts = sorted((root / "scenarios/scenario_meta").glob(f"part-{key}-*.parquet"))

    # ------------------------------------------------------------------------- the variant

    def at_state(self, state_id: str) -> tuple[ElectricalModel, ExtractionPlan]:
        """The model and the extraction plan at a network state (cached)."""
        if state_id not in self._models:
            if len(self._models) >= MAX_CACHED_STATES:
                self._models.pop(next(iter(self._models)))  # the oldest
            model = self.stored.model(state_id)
            self._models[state_id] = (model, build_plan(model))
        return self._models[state_id]

    def first_state(self) -> str | None:
        """The state of the first converged scenario, None if no scenario converged."""
        for path in self._meta_parts:
            meta = pq.read_table(path, columns=["pf_status", "state_id"])
            done = meta.filter(pc.equal(meta["pf_status"], pa.scalar("converged")))
            if done.num_rows:
                return str(done.column("state_id")[0].as_py())
        return None

    def variant(self) -> StoredVariant:
        """The variant as the view needs it (raises if no scenario converged)."""
        state = self.first_state()
        if state is None:
            raise ValueError(f"{self.variant_id}: no converged scenario")
        model, plan = self.at_state(state)
        names = [e.name for e in model.elements]
        return StoredVariant(
            variant_id=self.variant_id,
            model=model,
            plan=plan,
            customer_elements=tuple(names[j] for j in self._order("scenarios/loads")),
            pv_generators=tuple(names[j] for j in self._order("scenarios/der")),
            frequency_hz=self.frequency_hz,
            tolerance=self.tolerance,
            s_base_va=self.s_base_va,
            state_model=lambda sid: self.at_state(sid)[0],
        )

    def _order(self, table: str) -> list[int]:
        """Element indices of the first scenario with rows in ``table`` (every scenario of a
        variant has the same rows in the same order: its customers, its PV units)."""
        for meta in self._meta_parts:
            path = self.root / table / meta.name
            if not path.is_file():
                continue
            rows = pq.read_table(path, columns=["scenario_id", "element_idx"])
            if rows.num_rows:
                sids = rows.column("scenario_id").to_numpy()
                first = sids == sids[0]
                return [int(e) for e in rows.column("element_idx").to_numpy()[first]]
        return []

    # ------------------------------------------------------------------------ its scenarios

    def converged(self) -> Iterator[tuple[int, list[int]]]:
        """The ids of the converged scenarios of each chunk."""
        for meta_path in self._meta_parts:
            chunk = int(meta_path.stem.rsplit("-", 1)[1])
            meta = pq.read_table(meta_path, columns=["scenario_id", "pf_status"])
            done = meta.filter(pc.equal(meta["pf_status"], pa.scalar("converged")))
            yield chunk, [int(s) for s in done.column("scenario_id").to_numpy().tolist()]

    def chunks(self) -> Iterator[tuple[int, list[StoredScenario]]]:
        """The converged scenarios of each chunk, in ``scenario_id`` order."""
        for meta_path in self._meta_parts:
            chunk = int(meta_path.stem.rsplit("-", 1)[1])
            meta = pq.read_table(meta_path, columns=["scenario_id", "pf_status", "state_id"])
            done = meta.filter(pc.equal(meta["pf_status"], pa.scalar("converged")))
            if done.num_rows == 0:
                yield chunk, []
                continue
            yield chunk, list(self._scenarios(meta_path.name, done))

    def _scenarios(self, name: str, done: pa.Table) -> Iterator[StoredScenario]:
        results = pq.read_table(self.root / "scenarios/node_results" / name)
        results = results.sort_by([("scenario_id", "ascending"), ("node_idx", "ascending")])
        sids = done.column("scenario_id").to_numpy()
        states = [str(s) for s in done.column("state_id").to_pylist()]
        n_nodes = self.stored.tables[NODES].num_rows
        if results.num_rows != sids.size * n_nodes:
            raise ValueError(f"node_results/{name}: {results.num_rows} rows for {sids.size} "
                             f"converged scenarios of {n_nodes} nodes")  # fmt: skip
        stored = results.column("scenario_id").to_numpy().reshape(-1, n_nodes)[:, 0]
        if not np.array_equal(np.sort(sids), stored):
            raise ValueError(f"node_results/{name}: not the converged scenarios of the chunk")
        v_all = (results["v_re"].to_numpy() + 1j * results["v_im"].to_numpy()).reshape(-1, n_nodes)
        row_of = {int(s): r for r, s in enumerate(stored.tolist())}
        loads = _by_scenario(self.root / "scenarios/loads" / name, ["p_w", "q_var"])
        der = _by_scenario(self.root / "scenarios/der" / name, ["p_w", "q_var", "p_avail_w"])
        for sid, state in sorted(zip(sids.tolist(), states, strict=True)):
            v = v_all[row_of[int(sid)]]
            _, plan = self.at_state(state)
            p, q = loads.get(int(sid), (np.zeros(0), np.zeros(0)))
            pv = der.get(int(sid))
            yield StoredScenario(
                scenario_id=int(sid),
                variant_id=self.variant_id,
                state_id=state,
                v=np.asarray(v, dtype=np.complex128),
                i_pd=plan.currents(v),
                load_p_w=p,
                load_q_var=q,
                pv_p_w=None if pv is None else pv[0],
                pv_q_var=None if pv is None else pv[1],
                pv_p_avail_w=None if pv is None else pv[2],
            )


def _by_scenario(path: Path, columns: list[str]) -> dict[int, tuple[NDArray[np.float64], ...]]:
    """``columns`` of each scenario's rows of a part (float64), in the stored row order."""
    if not path.is_file():
        return {}
    table = pq.read_table(path, columns=["scenario_id", *columns])
    if table.num_rows == 0:  # e.g. a variant without PV
        return {}
    sids = table.column("scenario_id").to_numpy()
    order = np.argsort(sids, kind="stable")  # a scenario's rows keep their order
    sids = sids[order]
    values = [table.column(c).to_numpy().astype(np.float64)[order] for c in columns]
    unique, starts = np.unique(sids, return_index=True)
    stops = np.r_[starts[1:], sids.size]
    return {
        int(sid): tuple(v[a:b] for v in values)
        for sid, a, b in zip(unique.tolist(), starts.tolist(), stops.tolist(), strict=True)
    }
