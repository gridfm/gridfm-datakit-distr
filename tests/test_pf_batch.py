"""Tests for the per-scenario PF loop."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import SolveStatus
from gridfm_datakit_distr.model.state import NetworkState
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner, prepare_options
from gridfm_datakit_distr.variants.generate import VariantResult, generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from tests.circuits import CONTROLLED, VARIANT_FEEDER, write_circuit

SEED = 9
BASE: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["x"]},
    "scenarios": {"n_scenarios_per_variant": 10},
    "settings": {"seed": SEED, "residual_sample_rate": 1.0},
}


def config(**sections: Any) -> GenerationConfig:
    data = copy.deepcopy(BASE)
    for key, value in sections.items():
        data[key] = {**data.get(key, {}), **value}
    return GenerationConfig.model_validate(data)


def variant(
    tmp_path: Path, circuit: dict[str, str], cfg: GenerationConfig
) -> tuple[Network, VariantResult]:
    write_circuit(tmp_path / "src", circuit)
    network = Network(id="t/net", source="src", master="master.dss")
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options, tmp_path))
    result = generate_variant(
        network, targets, cfg.variants, SEED, 0, 0, options=options, root=tmp_path, der=cfg.der.pv
    )
    assert result.status == "ok"
    return network, result


def runner(tmp_path: Path, circuit: dict[str, str], cfg: GenerationConfig) -> VariantRunner:
    network, result = variant(tmp_path, circuit, cfg)
    return VariantRunner(network, result, cfg, 0, root=tmp_path)


def test_scenarios_converge_with_their_draws(tmp_path: Path) -> None:
    r = runner(tmp_path, CONTROLLED, config())
    outcomes = [r.run(s) for s in range(4)]
    for s, o in enumerate(outcomes):
        assert o.scenario_id == s
        assert o.pf_status == "converged", o.message
        assert o.v is not None
        assert o.v.shape == (len(r.model.nodes),)
        assert o.residual_rel is not None
        assert o.residual_rel < 1e-6
        assert 0.98 <= o.source_v_pu < 1.05
        assert o.state_id in r.registry.states
        assert o.profile_source == "synthetic"
        assert o.profile_time_s is None
    assert sum(r.registry.counts.values()) == 4


def test_results_do_not_depend_on_the_order(tmp_path: Path) -> None:
    cfg = config(controls={"tap_jitter_steps": 2})
    forward = runner(tmp_path / "a", CONTROLLED, cfg)
    backward = runner(tmp_path / "b", CONTROLLED, cfg)
    a = {s: forward.run(s) for s in range(6)}
    b = {s: backward.run(s) for s in reversed(range(6))}
    for s in range(6):
        assert a[s].state_id == b[s].state_id
        assert a[s].v is not None
        assert b[s].v is not None
        assert np.array_equal(a[s].v, b[s].v)  # bitwise


def test_without_controls_the_label_solve_alone_gives_the_same_results(tmp_path: Path) -> None:
    cfg = config(der={"pv": {"penetration": 1.0}})  # PV generators are set too
    fast = runner(tmp_path / "a", VARIANT_FEEDER, cfg)
    full = runner(tmp_path / "b", VARIANT_FEEDER, cfg)
    assert not fast._settles
    assert fast.pv is not None
    full._settles = True  # settle anyway: STATIC solves before the label solve
    for s in range(4):
        a, b = fast.run(s), full.run(s)
        assert (a.pf_status, a.state_id, a.origin) == (b.pf_status, b.state_id, b.origin)
        assert a.v is not None
        assert b.v is not None
        assert np.array_equal(a.v, b.v)  # bitwise


def test_a_non_convergent_scenario_is_recorded_without_results(tmp_path: Path) -> None:
    heavy = config(scenarios={"profiles": {"synthetic": {"global_scale": 300.0}}})  # 60 MW
    r = runner(tmp_path / "a", CONTROLLED, heavy)
    o = r.run(0)
    assert o.pf_status == "not_converged"
    assert o.v is None
    assert o.residual_rel is None
    assert o.pf_iterations > 0
    # the loop goes on, unaffected: a normal scenario after the failure equals a fresh one
    normal = runner(tmp_path / "b", CONTROLLED, config())
    r.config = normal.config
    r.loads = normal.loads
    after = r.run(1)
    fresh = runner(tmp_path / "c", CONTROLLED, config()).run(1)
    assert after.pf_status == fresh.pf_status == "converged"
    assert after.v is not None
    assert fresh.v is not None
    assert np.array_equal(after.v, fresh.v)


def test_a_state_whose_controls_fail_is_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry holds the states of the scenarios that reach the dataset; the failed scenario
    has neither a state nor an origin."""
    r = runner(tmp_path, CONTROLLED, config())
    state, _ = r._controls(0)
    monkeypatch.setattr(r, "_controls", lambda s: (state, SolveStatus(False, 7)))
    failed = r.run(0)
    assert (failed.pf_status, failed.state_id, failed.origin) == ("not_converged", None, None)
    assert not r.registry.states
    monkeypatch.undo()
    o = r.run(0)
    assert o.pf_status == "converged"
    assert o.state_id in r.registry.states
    assert o.origin == "static_settle"


def test_each_scenario_records_its_own_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenarios that share a state can reach it by different routes: the origin belongs to the
    scenario, not to the state."""
    r = runner(tmp_path, CONTROLLED, config())
    settled = r.run(0)
    real = r._controls

    def jittered(s: int) -> tuple[NetworkState, SolveStatus | None]:
        state, status = real(s)
        return dataclasses.replace(state, origin="static_settle+jitter"), status

    monkeypatch.setattr(r, "_controls", jittered)
    again = r.run(0)
    assert again.state_id == settled.state_id
    assert (settled.origin, again.origin) == ("static_settle", "static_settle+jitter")


def test_an_off_setpoint_scenario_has_no_results(tmp_path: Path) -> None:
    narrow = config(
        normalisation={"loads": {"vminpu": 0.995, "vmaxpu": 1.5}},
        controls={"mode": "as_is"},
        scenarios={"profiles": {"synthetic": {"global_scale": 10.0}}, "source_voltage_pu": 1.0},
    )
    o = runner(tmp_path, CONTROLLED, narrow).run(0)
    assert o.pf_status == "off_setpoint"
    assert o.n_off_setpoint >= 1
    assert o.v is None


def test_errors_are_recorded_and_the_loop_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = runner(tmp_path, CONTROLLED, config())

    def broken(*args: Any) -> Any:
        raise RuntimeError("provider down")

    monkeypatch.setattr(r.loads, "draw", broken)
    o = r.run(0)
    assert o.pf_status == "error"
    assert o.message == "RuntimeError: provider down"
    assert o.v is None
    monkeypatch.undo()
    assert r.run(1).pf_status == "converged"


def test_pv_units_get_their_outputs(tmp_path: Path) -> None:
    cfg = config(
        der={"pv": {"penetration": 1.0}},
        controls={"mode": "as_is"},
    )
    r = runner(tmp_path, VARIANT_FEEDER, cfg)
    assert r.pv is not None
    assert len(r.pv.units) == 6
    outcomes = [r.run(s) for s in range(8)]
    assert all(o.pf_status == "converged" for o in outcomes), [o.message for o in outcomes]
    assert any(o.pv_p_w is not None and o.pv_p_w.sum() > 0 for o in outcomes)
    assert all(o.der_penetration == 1.0 for o in outcomes)


def test_outcome_fields(tmp_path: Path) -> None:
    o: ScenarioOutcome = runner(tmp_path, CONTROLLED, config()).run(3)
    assert o.variant_id == "t/net/v0000"
    assert o.network_id == "t/net"
    assert o.seed > 0
    assert o.load_p_w.shape == o.load_q_var.shape == (1,)
    assert o.t_solve_s > 0
