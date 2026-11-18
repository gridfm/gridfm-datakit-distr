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
OPF's bounds and every rated transformer within its rating, ``B`` if not, ``C`` without a
power-flow label. The bounds are
``networks.voltage_limits`` in per unit of each phase node's nominal line-to-neutral voltage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.extract.source import read_source
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.model.ids import split_scenario_id
from gridfm_datakit_distr.opf.problem import (
    Costs,
    FloatArray,
    OPFProblem,
    OPFResult,
    PowerFlowStart,
    economic,
    power_base,
    vpn_bounds,
    winding_va,
)
from gridfm_datakit_distr.opf.support import unsupported_reason
from gridfm_datakit_distr.opf.verify import Verification, verify
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner
from gridfm_datakit_distr.solve.tables import concat_cast

ACTIVE_REL = 1e-6
"""A limit is counted active when the solution is within this relative distance of it."""
NO_FREE_VARIABLE = "no_free_variable"
"""Status of a class-B scenario whose PV units are all fixed: none placed, or none with available
power or a reactive range. Not solved: nothing can move to remove the violation."""
NO_FREE_MESSAGE = "PV units all fixed (none, or no power or reactive range): nothing can move"


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
    costs: Costs | None = None
    """The scenario's cost coefficients (source first)."""
    q_max_var: FloatArray | None = None
    """The largest |q| each PV unit could dispatch (var)."""
    active: tuple[int, int, int] | None = None
    """Limits active at the solution: voltages at their lower and upper bounds, transformers at
    their rating (within :data:`ACTIVE_REL` of the bound)."""
    start: str | None = None
    """``flat``, or ``power_flow`` when the flat start did not solve and the label comes from a
    second solve started at the scenario's power flow."""


class OPFLabeller:
    """The OPF labels of one runner's scenarios."""

    def __init__(self, runner: VariantRunner, config: GenerationConfig) -> None:
        self.runner = runner
        self.seed = config.settings.seed
        opf = config.labels.opf
        self.preset = opf.costs
        self.sampled = opf.sampled
        self.time_limit_s = opf.time_limit_s
        model = runner.model
        self.reason = unsupported_reason(model)
        """Why the network is outside the OPF's scope; None inside."""
        units = runner.pv.units if runner.pv is not None else ()
        self.units = units
        index = {e.name: e.idx for e in model.elements}
        self.pv_idx = np.array([index[u.generator] for u in units], dtype=np.int32)
        self.s_rated_va = np.array([u.s_rated_va for u in units], dtype=np.float64)
        limits = config.networks.voltage_limits
        self.limits_pu = (limits.vmin_pu, limits.vmax_pu)
        self.pairs, self.vpn_lo_v, self.vpn_hi_v = vpn_bounds(model, *self.limits_pu)
        self.source = None if self.reason is not None else read_source(runner.ctx, model)
        self.s_base_va = power_base([abs(complex(c.p_w, c.q_var)) for c in runner.customers])
        """Power base of the NLP's scaling, from the customers' nominal loads."""
        self._problems: dict[str, OPFProblem] = {}

    def label(self, o: ScenarioOutcome) -> OPFRow:
        """The OPF row of scenario ``o``, run right after ``runner.run`` returned it."""
        labelled = o.pf_status == "converged" and o.v is not None and o.state_id is not None
        opf_class = self._class(o.v, str(o.state_id)) if labelled and o.v is not None else "C"
        costs = self.costs_of(o.scenario_id)
        if self.reason is not None:
            return OPFRow(o.scenario_id, "not_supported", opf_class, self.reason, costs=costs)
        if not labelled:
            return OPFRow(o.scenario_id, "no_pf_label", opf_class, costs=costs)
        p_avail = o.pv_p_avail_w if o.pv_p_avail_w is not None else np.zeros(0)
        q_max = np.array([u.q_max_var(float(p)) for u, p in zip(self.units, p_avail, strict=True)])
        if opf_class == "B" and not np.any((p_avail > 0) | (q_max > 0)):
            # No free variable: the NLP is the power flow's square system, which Ipopt solves while
            # ignoring the bounds (`Feasible_Point_Found`); the violation stands by construction.
            return OPFRow(o.scenario_id, NO_FREE_VARIABLE, opf_class, NO_FREE_MESSAGE, costs=costs)
        try:
            problem = self._problem(str(o.state_id))

            def solve(start: PowerFlowStart | None = None) -> OPFResult:
                return problem.solve(
                    source_v_pu=o.source_v_pu, load_p_w=o.load_p_w, load_q_var=o.load_q_var,
                    pv_p_avail_w=p_avail, pv_s_rated_va=self.s_rated_va, costs=costs,
                    pv_q_max_var=q_max, start=start,
                )  # fmt: skip

            result = solve()
            start = "flat"
            if not result.solved:
                # A second solve from the power-flow solution, a feasible point when the scenario
                # is of class A: from the flat start Ipopt can end such a scenario infeasible or
                # unfinished (T3.4: IEEE 13, 34, 123).
                assert o.v is not None
                empty = np.zeros(len(self.units))
                pf = PowerFlowStart(
                    v=o.v,
                    pv_p_w=o.pv_p_w if o.pv_p_w is not None else empty,
                    pv_q_var=o.pv_q_var if o.pv_q_var is not None else empty,
                )
                retry = solve(pf)
                result = replace(retry, t_solve_s=result.t_solve_s + retry.t_solve_s)
                start = "power_flow"
            check = None
            if result.solved:
                check = verify(self.runner.ctx, self.runner.model, problem, result)
        except Exception as exc:  # recorded, never dropped
            first = str(exc).splitlines()[0] if str(exc) else ""
            message = f"{type(exc).__name__}: {first}"
            return OPFRow(o.scenario_id, "error", opf_class, message, costs=costs)
        p_pf = o.pv_p_w if o.pv_p_w is not None else np.zeros(0)
        curtailment = float(np.sum(p_pf - result.pv_p_w))
        active = self._active(problem, result, str(o.state_id)) if result.solved else None
        return OPFRow(
            o.scenario_id, result.status, opf_class, None, result, check, curtailment, costs,
            q_max, active, start,
        )  # fmt: skip

    def _active(
        self, problem: OPFProblem, result: OPFResult, state_id: str
    ) -> tuple[int, int, int]:
        """Counts of the voltage bounds and transformer ratings active at ``result``."""
        pairs, v = problem.pairs, result.v
        neutral = np.where(pairs[:, 1] == GROUND, 0.0, v[pairs[:, 1]])
        vpn = np.abs(v[pairs[:, 0]] - neutral)
        low = int(np.sum(vpn <= problem.vpn_lo_v * (1 + ACTIVE_REL)))
        high = int(np.sum(vpn >= problem.vpn_hi_v * (1 - ACTIVE_REL)))
        model = self.runner.state_model(state_id)
        names = set(problem.transformer_names)
        ratings = [(e, e.rating_va) for e in model.elements if e.name in names and e.rating_va]
        at_rating = sum(winding_va(e, v) >= va * (1 - ACTIVE_REL) for e, va in ratings)
        return low, high, int(at_rating)

    def costs_of(self, sid: int) -> Costs:
        """The cost coefficients of scenario ``sid``: the preset ``economic``, or ``sampled``
        drawn from the scenario's ``OPF_COSTS`` stream (source c1, c2, then c1 and c2 per unit)."""
        n = len(self.units)
        if self.preset == "economic":
            return economic(n)
        s = split_scenario_id(sid)[2]
        rng = scenario_rng(self.seed, self.runner.net, self.runner.k, s, ScenarioStream.OPF_COSTS)
        spec = self.sampled
        source = [float(spec.source_c1.sample(rng)), float(spec.source_c2.sample(rng))]
        c1 = [float(spec.pv_c1.sample(rng)) for _ in range(n)]
        c2 = [float(spec.pv_c2.sample(rng)) for _ in range(n)]
        return Costs(c1=np.array([source[0], *c1]), c2=np.array([source[1], *c2]))

    def _problem(self, state_id: str) -> OPFProblem:
        if state_id not in self._problems:
            assert self.source is not None
            model = self.runner.state_model(state_id)
            problem = OPFProblem(
                model, self.source, *self.limits_pu, self.time_limit_s, self.s_base_va
            )
            customers = tuple(c.element for c in self.runner.customers)
            units = self.runner.pv.units if self.runner.pv is not None else ()
            if problem.load_names != customers or problem.pv_names != tuple(
                u.generator for u in units
            ):
                raise ValueError("the OPF's loads or PV units differ from the scenario's")
            self._problems[state_id] = problem
        return self._problems[state_id]

    def _class(self, v: NDArray[np.complex128], state_id: str) -> str:
        """``A`` when the uncontrolled power flow keeps every voltage within its bounds and every
        rated transformer within its rating, ``B`` otherwise."""
        neutral = np.where(self.pairs[:, 1] == GROUND, 0.0, v[self.pairs[:, 1]])
        vpn = np.abs(v[self.pairs[:, 0]] - neutral)
        inside = bool(np.all((vpn >= self.vpn_lo_v) & (vpn <= self.vpn_hi_v)))
        model = self.runner.state_model(state_id)
        rated = all(
            winding_va(e, v) <= e.rating_va
            for e in model.elements
            if e.enabled and e.element_class == "transformer" and e.rating_va
        )
        return "A" if inside and rated else "B"

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
                "source_c1": float(r.costs.c1[0]) if r.costs is not None else None,
                "source_c2": float(r.costs.c2[0]) if r.costs is not None else None,
                "objective": None if r.result is None else r.result.objective,
                "source_p_w": None if r.result is None else r.result.source_p_w,
                "iterations": None if r.result is None else r.result.iterations,
                "t_solve_s": None if r.result is None else r.result.t_solve_s,
                "curtailment_w": r.curtailment_w,
                "verified": None if r.verification is None else r.verification.verified,
                "verify_reason": None if r.verification is None else r.verification.reason,
                "verify_max_dv_pu": None if r.verification is None else r.verification.max_dv_pu,
                "verify_max_bound_excess_pu": None
                if r.verification is None
                else r.verification.max_bound_excess_pu,
                "controls_hold": None if r.verification is None else r.verification.controls_hold,
                "n_vmin_active": None if r.active is None else r.active[0],
                "n_vmax_active": None if r.active is None else r.active[1],
                "n_rating_active": None if r.active is None else r.active[2],
                "opf_start": r.start,
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
            assert r.costs is not None
            assert r.q_max_var is not None
            der.append(
                pa.table(
                    {
                        "scenario_id": np.full(n_pv, r.scenario_id, dtype=np.int64),
                        "element_idx": self.pv_idx,
                        "p_w": r.result.pv_p_w,
                        "q_var": r.result.pv_q_var,
                        "q_max_var": r.q_max_var,
                        "c1": r.costs.c1[1:],
                        "c2": r.costs.c2[1:],
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
