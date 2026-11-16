"""The OPF problem on a CSIRO four-wire feeder with PV large enough to curtail."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.compile import compile_network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.opf.problem import SOLVED, OPFProblem, economic, read_source
from gridfm_datakit_distr.scenarios.profiles import customers_of

pytestmark = [pytest.mark.network_data]

NETWORK = "csiro_enwl/n05_f8"
PV_KW = 30.0
"""A unit at every fourth load (definition order), single phase, unity power factor."""
VPN_MIN_V, VPN_MAX_V = 216.2, 253.0
OBJECTIVE_W = -194192.36617879625
"""Derived number: the objective this formulation reached on this case in the study that chose
it (Ipopt tol 1e-9, OpenDSS's source model), 13.0 % of the PV curtailed."""


def pv_edits(network_id: str) -> list[str]:
    ctx = DssContext.new()
    compile_network(ctx, load_catalog().network(network_id))
    edits = []
    for k, name in enumerate(ctx.api.Loads.AllNames(), start=1):
        if k % 4 == 0:
            ctx.activate_element(f"Load.{name}")
            bus = ctx.api.CktElement.BusNames()[0]
            edits.append(
                f"New Generator.pv_{name} Bus1={bus} Phases=1 kV=0.23 kW={PV_KW} kvar=0 "
                "Model=1 vminpu=0.6 vmaxpu=1.5"
            )
    return edits


def test_the_objective_of_a_curtailing_case_is_reproduced(
    network_master: Callable[[str], Path],
) -> None:
    network_master(NETWORK)
    prepared = prepare_network(load_catalog().network(NETWORK), edits=pv_edits(NETWORK))
    problem = OPFProblem(
        prepared.model, read_source(prepared.ctx, prepared.model), VPN_MIN_V, VPN_MAX_V
    )
    customers = customers_of(prepared)
    assert tuple(c.element for c in customers) == problem.load_names
    n_pv = len(problem.pv_names)
    result = problem.solve(
        source_v_pu=1.0,
        load_p_w=[c.p_w for c in customers],
        load_q_var=[c.q_var for c in customers],
        pv_p_avail_w=[PV_KW * 1e3] * n_pv,
        pv_s_rated_va=[PV_KW * 1e3] * n_pv,
        costs=economic(n_pv),
        pv_q_max_var=[0.0] * n_pv,  # unity power factor, as in the reference case
    )
    assert result.status == SOLVED
    assert abs(result.objective - OBJECTIVE_W) <= 1e-6 * abs(OBJECTIVE_W)
    curtailment = 1.0 - result.pv_p_w.sum() / (PV_KW * 1e3 * n_pv)
    assert curtailment > 0.1
