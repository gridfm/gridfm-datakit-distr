"""OPF labels of a variant's scenarios, right after each scenario's power flow.

An :class:`OPFLabeller` serves one :class:`VariantRunner`. For a scenario whose power flow gave a
label it builds the :class:`OPFProblem` of the scenario's network state (once per state), solves it
from a flat start and, if solved, verifies the dispatch in OpenDSS in the runner's context, which
still holds the scenario. Every scenario gets a row (outputs.md §8):

- the Ipopt status of the solve (only ``Solve_Succeeded`` is a label);
- ``not_supported`` with the reason, for a network outside the OPF's scope (opf.md);
- ``no_pf_label`` for a scenario without a power-flow label (not solved);
- ``error`` with the exception, if building, solving or verifying raised.

``opf_class``: ``A`` if the uncontrolled power flow keeps every phase-to-neutral voltage within the
OPF's bounds, ``B`` if not, ``C`` without a power-flow label. The bounds are
``networks.voltage_limits`` in per unit of each phase node's nominal line-to-neutral voltage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.opf.problem import (
    Costs,
    OPFProblem,
    OPFResult,
    economic,
    read_source,
    vpn_bounds,
)
from gridfm_datakit_distr.opf.support import unsupported_reason
from gridfm_datakit_distr.opf.verify import Verification, verify
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner
from gridfm_datakit_distr.solve.tables import concat_cast


@dataclass(frozen=True, slots=True, eq=False)
class OPFRow:
    """The OPF of one scenario."""

    scenario_id: int
    status: str
    opf_class: str
    message: str | None = None
    result: OPFResult | None = None
    verification: Verification | None = None
    curtailment_w: float | None = None
    """``sum(p_pf - p_opf)`` over the PV units (W), against the uncontrolled dispatch."""


class OPFLabeller:
    """The OPF labels of one runner's scenarios."""

    def __init__(self, runner: VariantRunner, config: GenerationConfig) -> None:
        self.runner = runner
        opf = config.labels.opf
        self.preset = opf.costs
        self.time_limit_s = opf.time_limit_s
        model = runner.model
        self.reason = unsupported_reason(model)
        """Why the network is outside the OPF's scope; None inside."""
        units = runner.pv.units if runner.pv is not None else ()
        index = {e.name: e.idx for e in model.elements}
        self.pv_idx = np.array([index[u.generator] for u in units], dtype=np.int32)
        self.s_rated_va = np.array([u.s_rated_va for u in units], dtype=np.float64)
        self.costs: Costs = economic(len(units))
        limits = config.networks.voltage_limits
        self.limits_pu = (limits.vmin_pu, limits.vmax_pu)
        self.pairs, self.vpn_lo_v, self.vpn_hi_v = vpn_bounds(model, *self.limits_pu)
        self.source = None if self.reason is not None else read_source(runner.ctx, model)
        self._problems: dict[str, OPFProblem] = {}

    def label(self, o: ScenarioOutcome) -> OPFRow:
        """The OPF row of scenario ``o``, run right after ``runner.run`` returned it."""
        labelled = o.pf_status == "converged" and o.v is not None and o.state_id is not None
        opf_class = self._class(o.v) if labelled and o.v is not None else "C"
        if self.reason is not None:
            return OPFRow(o.scenario_id, "not_supported", opf_class, self.reason)
        if not labelled:
            return OPFRow(o.scenario_id, "no_pf_label", opf_class)
        try:
            problem = self._problem(str(o.state_id))
            empty = np.zeros(0)
            p_avail = o.pv_p_avail_w if o.pv_p_avail_w is not None else empty
            result = problem.solve(
                source_v_pu=o.source_v_pu,
                load_p_w=o.load_p_w,
                load_q_var=o.load_q_var,
                pv_p_avail_w=p_avail,
                pv_s_rated_va=self.s_rated_va,
                costs=self.costs,
            )
            check = None
            if result.solved:
                check = verify(self.runner.ctx, self.runner.model, problem, result)
        except Exception as exc:  # recorded, never dropped
            first = str(exc).splitlines()[0] if str(exc) else ""
            return OPFRow(o.scenario_id, "error", opf_class, f"{type(exc).__name__}: {first}")
        p_pf = o.pv_p_w if o.pv_p_w is not None else np.zeros(0)
        curtailment = float(np.sum(p_pf - result.pv_p_w))
        return OPFRow(o.scenario_id, result.status, opf_class, None, result, check, curtailment)

    def _problem(self, state_id: str) -> OPFProblem:
        if state_id not in self._problems:
            assert self.source is not None
            model = self.runner.state_model(state_id)
            problem = OPFProblem(model, self.source, *self.limits_pu, self.time_limit_s)
            customers = tuple(c.element for c in self.runner.customers)
            units = self.runner.pv.units if self.runner.pv is not None else ()
            if problem.load_names != customers or problem.pv_names != tuple(
                u.generator for u in units
            ):
                raise ValueError("the OPF's loads or PV units differ from the scenario's")
            self._problems[state_id] = problem
        return self._problems[state_id]

    def _class(self, v: NDArray[np.complex128]) -> str:
        """``A`` within the bounds, ``B`` outside."""
        neutral = np.where(self.pairs[:, 1] == GROUND, 0.0, v[self.pairs[:, 1]])
        vpn = np.abs(v[self.pairs[:, 0]] - neutral)
        inside = (vpn >= self.vpn_lo_v) & (vpn <= self.vpn_hi_v)
        return "A" if bool(np.all(inside)) else "B"

    def tables(self, rows: Sequence[OPFRow], precision: schema.Precision) -> dict[str, pa.Table]:
        """``opf/opf_meta`` for every row; ``opf/opf_der`` and ``opf/opf_node_results`` for the
        solved scenarios."""
        meta = [
            {
                "scenario_id": r.scenario_id,
                "opf_status": r.status,
                "opf_message": r.message,
                "opf_class": r.opf_class,
                "cost_preset": self.preset,
                "source_c1": float(self.costs.c1[0]),
                "source_c2": float(self.costs.c2[0]),
                "objective": None if r.result is None else r.result.objective,
                "iterations": None if r.result is None else r.result.iterations,
                "t_solve_s": None if r.result is None else r.result.t_solve_s,
                "curtailment_w": r.curtailment_w,
                "verified": None if r.verification is None else r.verification.verified,
                "verify_reason": None if r.verification is None else r.verification.reason,
                "verify_max_dv_pu": None if r.verification is None else r.verification.max_dv_pu,
                "verify_max_bound_excess_pu": None
                if r.verification is None
                else r.verification.max_bound_excess_pu,
            }
            for r in rows
        ]
        out = {"opf/opf_meta": pa.Table.from_pylist(meta, schema=schema.opf_meta(precision))}
        solved = [r for r in rows if r.result is not None and r.result.solved]
        n_pv = self.pv_idx.size
        n_nodes = self.runner.model.n_nodes
        der, nodes = [], []
        for r in solved:
            assert r.result is not None
            der.append(
                pa.table(
                    {
                        "scenario_id": np.full(n_pv, r.scenario_id, dtype=np.int64),
                        "element_idx": self.pv_idx,
                        "p_w": r.result.pv_p_w,
                        "q_var": r.result.pv_q_var,
                        "c1": self.costs.c1[1:],
                        "c2": self.costs.c2[1:],
                    }
                )
            )
            nodes.append(
                pa.table(
                    {
                        "scenario_id": np.full(n_nodes, r.scenario_id, dtype=np.int64),
                        "node_idx": np.arange(n_nodes, dtype=np.int32),
                        "v_re": r.result.v.real,
                        "v_im": r.result.v.imag,
                    }
                )
            )
        out["opf/opf_der"] = concat_cast(der, schema.opf_der(precision))
        out["opf/opf_node_results"] = concat_cast(nodes, schema.node_results(precision))
        return out
