"""A negative-sequence component of the source voltage (``scenarios.source_unbalance``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import DssContext, complex_from_interleaved
from gridfm_datakit_distr.extract.injections import pc_source_admittance
from gridfm_datakit_distr.extract.source import (
    A,
    Source,
    negative_sequence,
    read_source,
    scenario_e,
    vsource_of,
)
from gridfm_datakit_distr.io import datakit_columns as datakit
from gridfm_datakit_distr.io.reader import DatasetReader
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, add_negative_sequence_source
from gridfm_datakit_distr.validate.closed_loop import closed_loop
from gridfm_datakit_distr.validate.consistency import compare_admittance
from gridfm_datakit_distr.views.kron import KronView
from gridfm_datakit_distr.views.posseq_export import export_posseq
from tests.circuits import CONTROLLED, FOUR_WIRE, VARIANT_FEEDER, opf_catalog, write_circuit
from tests.test_pf_batch import config, runner, variant

UNBALANCE = {"distribution": "uniform", "low": 0.005, "high": 0.02}
SINGLE_PHASE = {
    "master.dss": """\
Clear
New Circuit.one basekv=0.23 pu=1.0 phases=1 bus1=src.1 MVAsc1=10
New Line.l bus1=src.1 bus2=b.1 phases=1 r1=0.1 x1=0.05 r0=0.1 x0=0.05 c1=0 c0=0 length=1
New Load.h bus1=b.1 phases=1 kV=0.23 kW=2 kvar=0.5 model=1
Set VoltageBases=[0.23]
CalcVoltageBases
""",
}
READBACK_REL = 1e-9
"""The engine realises a phase angle to about 1.1e-10 rad per 120 degrees (its Vsource as its
Isource, spike s58): the floor of every readback of the source."""


def sequences(v: np.ndarray) -> tuple[complex, complex, complex]:
    """Zero, positive and negative sequence of an a-b-c vector."""
    return (
        complex(v.sum() / 3),
        complex((v[0] + A * v[1] + A**2 * v[2]) / 3),
        complex((v[0] + A**2 * v[1] + A * v[2]) / 3),
    )


def thevenin(ctx: DssContext, model: ElectricalModel, source: Source) -> np.ndarray:
    """The ideal source voltages behind ``Y_s`` read back from a solved circuit:
    ``V_s + Z_s I_s``, ``I_s`` the current the Vsource and any Isource push into the network."""
    v = ctx.ynode_voltages()[ctx.ynode_positions(model.nodes[int(n)].opendss_node
                                                 for n in source.nodes)]  # fmt: skip
    vs = vsource_of(model)
    k = [vs.terminals[0].index(int(n)) for n in source.nodes]
    ctx.activate_element(vs.name)
    i = -complex_from_interleaved(ctx.api.CktElement.Currents())[k]
    j = ctx.api.Isource.First()
    while j > 0:
        ctx.activate_element(f"Isource.{ctx.api.Isource.Name()}")
        i -= complex_from_interleaved(ctx.api.CktElement.Currents())[:3]
        j = ctx.api.Isource.Next()
    return np.asarray(v + np.linalg.solve(source.y_s, i))


def unbalanced(**sections: Any) -> GenerationConfig:
    return config(scenarios={"source_unbalance": UNBALANCE}, **sections)


# --------------------------------------------------------------------------------- E


def test_the_negative_sequence_has_the_drawn_ratio_and_angle() -> None:
    e_v = 230.0 * np.exp(1j * np.radians(30.0 - 120.0 * np.arange(3)))
    np.testing.assert_array_equal(scenario_e(e_v, 1.03), 1.03 * e_v)  # balanced: exactly
    np.testing.assert_array_equal(scenario_e(e_v, 1.03, 0.0, 2.0), 1.03 * e_v)
    e = scenario_e(e_v, 1.03, 0.02, 2.0)
    zero, pos, neg = sequences(e)
    assert pos == pytest.approx(1.03 * 230.0 * np.exp(1j * np.radians(30.0)), rel=1e-14)
    assert neg / pos == pytest.approx(0.02 * np.exp(2j), rel=1e-12)
    assert abs(zero) < 1e-12 * abs(pos)
    n = negative_sequence(e_v)
    np.testing.assert_allclose(np.abs(n), np.abs(e_v), rtol=1e-15)
    assert n[0] == e_v[0]


def test_a_negative_sequence_needs_three_phases() -> None:
    with pytest.raises(ValueError, match="three-phase"):
        scenario_e(np.array([230.0 + 0j]), 1.0, 0.01, 0.0)


# ----------------------------------------------------------------------- power flow


def test_the_default_adds_nothing(tmp_path: Path) -> None:
    r = runner(tmp_path, CONTROLLED, config())
    assert r.unbalance is None
    assert r.ctx.api.Isource.Count() == 0
    o = r.run(0)
    assert (o.pf_status, o.source_u2, o.source_u2_angle_rad) == ("converged", 0.0, None)


@pytest.mark.parametrize("circuit", [CONTROLLED, FOUR_WIRE], ids=["regulator", "four_wire"])
def test_the_drawn_unbalance_is_the_source_s(tmp_path: Path, circuit: dict[str, str]) -> None:
    """Each scenario's ideal source voltages, read back from the engine's solution, are
    ``scenario_e`` of its draws; the model keeps its nodes and elements (one source); the KCL
    residual, the admittance cross-check and the setpoint check hold."""
    balanced = runner(tmp_path / "a", circuit, config())
    r = runner(tmp_path / "b", circuit, unbalanced())
    assert r.unbalance is not None
    assert r.ctx.api.Isource.Count() == 1
    assert r.model.nodes == balanced.model.nodes
    assert [e.name for e in r.model.elements] == [e.name for e in balanced.model.elements]
    source = read_source(r.ctx, r.model)
    for s in range(4):
        o = r.run(s)
        assert o.pf_status == "converged", o.message  # setpoints met
        assert o.residual_rel is not None
        assert o.residual_rel < 1e-6
        assert 0.005 <= o.source_u2 < 0.02
        assert o.source_u2_angle_rad is not None
        assert 0.0 <= o.source_u2_angle_rad < 2 * np.pi
        e = scenario_e(source.e_v, o.source_v_pu, o.source_u2, o.source_u2_angle_rad)
        got = thevenin(r.ctx, r.model, source)
        assert np.abs(got - e).max() <= READBACK_REL * np.abs(e).max()
        _, pos, neg = sequences(got)
        assert abs(neg / pos) == pytest.approx(o.source_u2, abs=READBACK_REL)
    y = r.state_model(str(o.state_id))
    c3 = compare_admittance(y, y.ybus(), r.ctx.system_y(), pc_source_admittance(r.ctx, y))
    assert c3.passes(), c3


def test_at_no_load_the_source_bus_has_the_drawn_unbalance(tmp_path: Path) -> None:
    """Without current through the source impedance the bus voltage is the ideal one: its
    negative-sequence ratio and angle are the drawn ones (with load they are not, by ``Z_s I``)."""
    r = runner(tmp_path, FOUR_WIRE, unbalanced())
    o = r.run(1)
    assert o.pf_status == "converged", o.message
    api = r.ctx.api
    for cls in (api.Loads, api.Generators):  # the customers and the variant's PV units
        for name in cls.AllNames():
            cls.Name(name)
            cls.kW(0.0)
            cls.kvar(0.0)
    r.ctx.command("Set Mode=Snap")
    assert r.ctx.solve().converged
    source = read_source(r.ctx, r.model)
    v = r.ctx.ynode_voltages()[r.ctx.ynode_positions(r.model.nodes[int(n)].opendss_node
                                                     for n in source.nodes)]  # fmt: skip
    _, pos, neg = sequences(v)
    assert o.source_u2_angle_rad is not None
    assert neg / pos == pytest.approx(
        o.source_u2 * np.exp(1j * o.source_u2_angle_rad), abs=READBACK_REL
    )
    assert abs(pos) == pytest.approx(o.source_v_pu * abs(source.e_v[0]), rel=READBACK_REL)


def test_a_single_phase_source_is_refused(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", SINGLE_PHASE)
    prepared = prepare_network(Network(id="t/one", source="src", master="master.dss"),
                               root=tmp_path)  # fmt: skip
    with pytest.raises(ValueError, match="three-phase"):
        add_negative_sequence_source(prepared.ctx, prepared.model)
    assert prepared.ctx.api.Isource.Count() == 0


def test_the_kron_view_gets_the_same_source(tmp_path: Path) -> None:
    cfg = unbalanced(controls={"mode": "as_is"})
    network, result = variant(tmp_path, VARIANT_FEEDER, cfg)
    r = VariantRunner(network, result, cfg, 0, root=tmp_path)
    view = KronView(r, network, result, root=tmp_path)
    assert not view.identity
    o = r.run(2)
    assert o.pf_status == "converged", o.message
    k = view.solve(o, r.registry.states.get(o.state_id or ""))
    assert k.status == "converged", k.message
    source = read_source(view.ctx, view.model)
    e = scenario_e(source.e_v, o.source_v_pu, o.source_u2, o.source_u2_angle_rad)
    got = thevenin(view.ctx, view.model, source)
    assert np.abs(got - e).max() <= READBACK_REL * np.abs(e).max()


# ---------------------------------------------------------------------------- dataset

DATASET: dict[str, Any] = {
    "dataset_name": "b2",
    "networks": {"include": ["t/lv", "t/mvlv"]},
    "variants": {"n_variants": 1},
    "der": {"pv": {"penetration": 1.0, "kwp": 10.0}},
    "scenarios": {"n_scenarios_per_variant": 6},
    "views": {"kron": True},
    "labels": {"opf": {"enabled": True}},
    "settings": {"seed": 3, "num_processes": 1},
}


@pytest.fixture(scope="module")
def datasets(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    pytest.importorskip("casadi")
    root = tmp_path_factory.mktemp("b2")
    out = {}
    for name, spec in (("balanced", 0.0), ("unbalanced", UNBALANCE)):
        data = copy.deepcopy(DATASET)
        data["scenarios"]["source_unbalance"] = spec
        out[name] = root / name
        generate(GenerationConfig.model_validate(data), out[name], catalog=opf_catalog(root),
                 root=root)  # fmt: skip
    return out["balanced"], out["unbalanced"]


def table(out: Path, name: str) -> Any:
    return pq.read_table(out / name)


def test_the_unbalance_changes_the_source_and_nothing_else(datasets: tuple[Path, Path]) -> None:
    balanced, unbalanced_ = datasets
    for name in ("variants/variants", "variants/nodes", "variants/elements", "variants/source",
                 "variants/source_yprim", "scenarios/loads", "scenarios/der"):  # fmt: skip
        assert table(balanced, name).equals(table(unbalanced_, name)), name
    a, b = table(balanced, "scenarios/scenario_meta"), table(unbalanced_, "scenarios/scenario_meta")
    assert a["source_v_pu"].equals(b["source_v_pu"])
    assert set(a["source_u2"].to_pylist()) == {0.0}
    assert set(a["source_u2_angle_rad"].to_pylist()) == {None}
    u2 = np.array(b["source_u2"].to_pylist())
    angle = np.array(b["source_u2_angle_rad"].to_pylist())
    assert np.all((u2 >= 0.005) & (u2 < 0.02))
    assert np.all((angle >= 0.0) & (angle < 2 * np.pi))
    assert set(b["pf_status"].to_pylist()) == {"converged"}
    va, vb = table(balanced, "scenarios/node_results"), table(unbalanced_, "scenarios/node_results")
    assert not va.equals(vb)


def test_the_stored_tables_reproduce_the_unbalanced_voltages(datasets: tuple[Path, Path]) -> None:
    reader = DatasetReader(datasets[1])
    ids = reader.meta.column("scenario_id").to_pylist()
    for sc in reader.scenarios(ids):
        assert sc.source_u2 is not None
        assert sc.source_u2 > 0
        assert closed_loop(sc).max_dv_pu <= 1e-6


def test_the_opf_with_an_unbalanced_source_is_verified(datasets: tuple[Path, Path]) -> None:
    """The OPF sees the scenario's unbalanced source: its dispatch gives OpenDSS, which holds the
    same source, the OPF's voltages within 1e-6 pu."""
    opf = table(datasets[1], "opf/opf_meta").to_pylist()
    solved = [r for r in opf if r["opf_status"] == "Solve_Succeeded"]
    assert len(solved) >= len(opf) // 2
    assert all(r["verified"] and r["verify_max_dv_pu"] <= 1e-6 for r in solved)


def test_the_positive_sequence_view_keeps_v1_at_its_source(
    datasets: tuple[Path, Path], tmp_path: Path
) -> None:
    """The view's source is at ``|V1|`` of the full model's source bus, not at a phase's
    ``|V|``."""
    out = datasets[1]
    [result] = export_posseq(out, tmp_path / "export", networks=["t/lv"])
    bus = pq.read_table(result.raw_dir / datakit.FILES["bus"] / "scenario_partition=0")
    ref = bus.filter(pc.equal(bus["REF"], 1)).to_pylist()
    assert ref
    source = table(out, "variants/source").to_pylist()
    nodes = {r["node_idx"]: r for r in table(out, "variants/nodes").to_pylist()
             if r["variant_id"] == "t/lv/v0000"}  # fmt: skip
    idx = [r["node_idx"] for r in sorted(source, key=lambda r: r["position"])
           if r["variant_id"] == "t/lv/v0000"]  # fmt: skip
    base = nodes[idx[0]]["kv_base_ln"] * 1e3
    v = table(out, "scenarios/node_results").to_pylist()
    gap = 0.0
    for row in ref:
        phases = {r["node_idx"]: complex(r["v_re"], r["v_im"])
                  for r in v if r["scenario_id"] == row["x_scenario_id"]}  # fmt: skip
        _, pos, _ = sequences(np.array([phases[i] for i in idx]))
        assert row["Vm"] == pytest.approx(abs(pos) / base, rel=1e-9)
        gap = max(gap, abs(abs(phases[idx[0]]) / base - row["Vm"]))
    assert gap > 1e-4  # a phase's magnitude would differ: the check discriminates
