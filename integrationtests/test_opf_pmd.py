"""Cross-check of the OPF against PowerModelsDistribution (IVREN) on small CSIRO feeders, with the
PV reactive power free within a bound (Julia; run with DISTR_RUN_JULIA=1).

Both solve the same problem: PMD's model of the feeder, read by its own parser, whose source is a
four-conductor source with the phase impedances repeated on the neutral; the own formulation is
given that source. PV units at every fourth load, reactive power free within ±44 % of their peak
power (PMD's generator bounds; no capability circle on either side), phase-to-neutral voltages
within 216.2 to 253.0 V, cost `economic`. The objectives must agree within 1e-6 relative.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.source import Source, read_source
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.compile import compile_network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.opf.problem import SOLVED, OPFProblem, economic
from gridfm_datakit_distr.scenarios.profiles import customers_of

pytestmark = [pytest.mark.julia, pytest.mark.network_data, pytest.mark.slow]

PV_KW = 30.0
Q_FRACTION = 0.44
JULIA = r"""
function opf_pmd(path::String, vmin_kv::Float64, vmax_kv::Float64, qmax::Float64)
    PMD = PowerModelsDistribution
    eng = PMD.parse_file(path; transformations=[PMD.transform_loops!])
    eng["settings"]["sbase_default"] = 1.0
    vbase = first(values(eng["settings"]["vbases_default"]))
    PMD.add_bus_pn_pp_ng_vbounds!(eng, [1, 2, 3], 4;
                                  pn_lb_pu=vmin_kv / vbase, pn_ub_pu=vmax_kv / vbase)
    math = PMD.transform_data_model(eng; kron_reduce=false, phase_project=false)
    for (_, gen) in math["gen"]
        if startswith(gen["name"], "pv_")
            gen["cost"] = [0.0, 0.0]
            gen["qmin"] = fill(-qmax, length(gen["qmin"]))
            gen["qmax"] = fill(qmax, length(gen["qmax"]))
        else
            gen["cost"] = [1.0, 0.0]
        end
    end
    PMD.add_start_voltage!(math; coordinates=:rectangular, epsilon=0)
    opt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-9,
                                        "max_iter" => 3000, "max_cpu_time" => 300.0)
    pm = PMD.instantiate_mc_model(math, PMD.IVRENPowerModel, PMD.build_mc_opf)
    res = PMD._IM.optimize_model!(pm; optimizer=opt)
    return string(res["termination_status"]), res["objective"]
end
"""


def pv_loads(network_id: str) -> list[tuple[str, str]]:
    """(load, bus spec) of every fourth load in definition order, read from the engine."""
    ctx = DssContext.new()
    compile_network(ctx, load_catalog().network(network_id))
    out = []
    for k, name in enumerate(ctx.api.Loads.AllNames(), start=1):
        if k % 4 == 0:
            ctx.activate_element(f"Load.{name}")
            out.append((name, str(ctx.api.CktElement.BusNames()[0])))
    return out


def pmd_source(source: Source, model: object) -> Source:
    """PMD's source: the 3x3 impedance repeated on a fourth conductor to the source bus's neutral,
    with E = 0 on it."""
    z3 = np.linalg.inv(source.y_s)
    zs, zm = z3[0, 0], z3[0, 1]
    z4 = zm + (zs - zm) * np.eye(4)
    nodes = model.nodes  # type: ignore[attr-defined]
    bus = nodes[int(source.nodes[0])].bus
    (neutral,) = [n.idx for n in nodes if n.bus == bus and n.conductor == "n"]
    return Source(
        nodes=np.r_[source.nodes, neutral].astype(np.int64),
        y_s=np.linalg.inv(z4),
        e_v=np.r_[source.e_v, 0.0],
    )


@pytest.fixture(scope="module")
def julia() -> object:
    from gridfm_datakit_distr.opf.julia_env import load_packages

    jl = load_packages()
    jl.PowerModelsDistribution.silence_b()
    jl.seval("import JuMP")
    jl.seval(JULIA)
    return jl


@pytest.mark.parametrize(
    "network_id",
    [
        "csiro_enwl/n14_f6",
        "csiro_enwl/n05_f8",
        # with the reactive power free the two solvers stop at different local optima here: the
        # own objective is 2.2 kW (6.0e-3 relative) below PMD's, curtailing 21.1 kW against 24.0
        # (measured 2026-11-17; with the reactive power at zero they agreed to 1.6e-9, s36)
        pytest.param(
            "csiro_enwl/n21_f3", marks=pytest.mark.xfail(strict=True, reason="local optima")
        ),
    ],
)
def test_the_own_opf_matches_pmd_with_free_reactive_power(
    julia: object, network_master: Callable[[str], Path], network_id: str, tmp_path: Path
) -> None:
    master = network_master(network_id)
    loads = pv_loads(network_id)
    commands = [
        f"New Generator.pv_{name} Bus1={bus} Phases=1 kV=0.23 kW={PV_KW} kvar=0 Model=1 "
        "vminpu=0.6 vmaxpu=1.5"
        for name, bus in loads
    ]
    copy = tmp_path / "feeder"
    shutil.copytree(master.parent, copy)
    (copy / "master_pv.dss").write_text(
        master.read_text() + "\n" + "\n".join(commands) + "\n", encoding="utf-8"
    )
    pmd = julia.opf_pmd  # type: ignore[attr-defined]
    status, objective_kw = pmd(str(copy / "master_pv.dss"), 0.2162, 0.253, Q_FRACTION * PV_KW)
    assert str(status) == "LOCALLY_SOLVED"

    prepared = prepare_network(load_catalog().network(network_id), edits=commands)
    model = prepared.model
    source = pmd_source(read_source(prepared.ctx, model), model)
    problem = OPFProblem(model, source, 0.94, 1.10)
    customers = customers_of(prepared)
    n = len(problem.pv_names)
    result = problem.solve(
        source_v_pu=1.0,
        load_p_w=[c.p_w for c in customers],
        load_q_var=[c.q_var for c in customers],
        pv_p_avail_w=[PV_KW * 1e3] * n,
        pv_s_rated_va=[1e9] * n,  # no capability circle: PMD's IVREN generators have none
        costs=economic(n),
        pv_q_max_var=[Q_FRACTION * PV_KW * 1e3] * n,
    )
    assert result.status == SOLVED
    reference = float(objective_kw) * 1e3
    assert abs(result.objective - reference) <= 1e-6 * max(abs(reference), 1e3)
