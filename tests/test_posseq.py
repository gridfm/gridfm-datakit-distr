"""Tests for the positive-sequence view."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import complex_from_interleaved
from gridfm_datakit_distr.io import datakit_columns as datakit
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.normalize.bases import s_base_va
from gridfm_datakit_distr.solve.pf_batch import ScenarioOutcome, VariantRunner, prepare_options
from gridfm_datakit_distr.validate.balance import power_balance
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from gridfm_datakit_distr.views import posseq
from gridfm_datakit_distr.views.datakit_layout import ViewNetwork, compose
from gridfm_datakit_distr.views.posseq import (
    A1,
    ModelIndex,
    PosSeqView,
    find_backbone,
    projection,
    transformer_params,
)
from gridfm_datakit_distr.views.stored import StoredScenario, StoredVariant
from tests.circuits import VARIANT_FEEDER, write_circuit

CONFIG: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["x"]},
    "scenarios": {"n_scenarios_per_variant": 4},
    "labels": {"element_currents": True},  # the runner's currents stand for the stored ones
    "settings": {"seed": 5},
}

# A bank of single-phase regulators with unequal taps (no controller), a three-phase capacitor on
# the backbone and a single-phase lateral.
TAPS = (1.05, 1.025, 1.0)
BANK = {
    "master.dss": f"""\
Clear
New Circuit.bank basekv=12.47 pu=1.0 phases=3 bus1=src angle=0 MVAsc3=200 MVAsc1=200
New Transformer.ra phases=1 windings=2 buses=[src.1 r.1] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[500 500] xhl=1 %rs=[0.1 0.1] taps=[1 {TAPS[0]}] bank=reg
New Transformer.rb phases=1 windings=2 buses=[src.2 r.2] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[500 500] xhl=1 %rs=[0.1 0.1] taps=[1 {TAPS[1]}] bank=reg
New Transformer.rc phases=1 windings=2 buses=[src.3 r.3] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[500 500] xhl=1 %rs=[0.1 0.1] taps=[1 {TAPS[2]}] bank=reg
New Line.l1 bus1=r.1.2.3 bus2=a.1.2.3 phases=3 r1=0.3 x1=0.6 r0=0.6 x0=1.2 c1=10 c0=5
~ length=1 units=km
New Capacitor.cap bus1=a.1.2.3 phases=3 kvar=300 kv=12.47
New Line.lat bus1=a.1 bus2=x.1 phases=1 r1=0.4 x1=0.4 r0=0.4 x0=0.4 c1=0 c0=0
~ length=0.5 units=km
New Load.la bus1=a.1.2.3 phases=3 kV=12.47 kW=800 kvar=300 model=1
New Load.lx bus1=x.1 phases=1 kV=7.2 kW=200 kvar=50 model=1
Set VoltageBases=[12.47]
CalcVoltageBases
""",
}


def network(tmp_path: Path, circuit: dict[str, str]) -> Network:
    write_circuit(tmp_path / "src", circuit)
    return Network(id="t/net", source="src", master="master.dss")


def runner_of(
    tmp_path: Path, circuit: dict[str, str], **sections: Any
) -> tuple[VariantRunner, GenerationConfig]:
    data = copy.deepcopy(CONFIG)
    data.update(sections)
    cfg = GenerationConfig.model_validate(data)
    net = network(tmp_path, circuit)
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(net, options, tmp_path))
    result = generate_variant(
        net, targets, cfg.variants, 5, 0, options=options, root=tmp_path, der=cfg.der.pv
    )
    return VariantRunner(net, result, cfg, 0, root=tmp_path), cfg


def view_of(runner: VariantRunner, cfg: GenerationConfig) -> PosSeqView:
    """The view of a fresh runner's variant, given what the export reads of a dataset: the same
    model, orders and bases (``views/stored.py``; ``test_stored.py`` checks the reading)."""
    explicit = cfg.normalisation.s_base_va
    s_base = s_base_va(runner.ctx, runner.model)[0] if isinstance(explicit, str) else explicit
    source = StoredVariant(
        variant_id=runner.variant_id,
        model=runner.model,
        plan=runner.plan,
        customer_elements=tuple(c.element for c in runner.customers),
        pv_generators=tuple(u.generator for u in (runner.pv.units if runner.pv else ())),
        frequency_hz=float(runner.ctx.api.Solution.Frequency()),
        tolerance=runner.options.tolerance,
        s_base_va=float(s_base),
        state_model=runner.state_model,
    )
    return PosSeqView(source, cfg)


def stored(o: ScenarioOutcome) -> StoredScenario:
    """A converged outcome as the export reads it back."""
    assert o.pf_status == "converged", o.message
    assert o.v is not None
    assert o.i_pd is not None
    assert o.state_id is not None
    return StoredScenario(
        o.scenario_id, o.variant_id, o.state_id, o.v, o.i_pd, o.load_p_w, o.load_q_var,
        o.pv_p_w, o.pv_q_var, o.pv_p_avail_w,
    )  # fmt: skip


# ------------------------------------------------------------------------------ sequence algebra


def test_projection_is_the_positive_sequence_self_term() -> None:
    zs, zm = 0.4 + 0.9j, 0.1 + 0.4j
    circulant = np.full((3, 3), zm) + np.eye(3) * (zs - zm)
    assert projection(circulant)[0, 0] == pytest.approx(zs - zm, abs=1e-15)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    sym = m + m.T  # an untransposed line: mean(diag) - mean(offdiag)
    off = sym[~np.eye(3, dtype=bool)]
    assert projection(sym)[0, 0] == pytest.approx(np.diag(sym).mean() - off.mean(), abs=1e-14)
    # a common voltage on all phases (a neutral shift) has no positive sequence
    assert abs(A1.conj() @ np.ones(3)) < 1e-15


# ------------------------------------------------------------------------------------- backbone


def test_backbone_and_mapping(tmp_path: Path) -> None:
    model = prepare_network(network(tmp_path, VARIANT_FEEDER), root=tmp_path).model
    bb = find_backbone(model)
    assert bb.source == "mv"
    assert bb.buses == ("mv", "lv", "b1", "b2", "b3")
    assert bb.owner["b4"] == "b2"  # the two-wire lateral is aggregated where it starts
    kinds = [(g.from_bus, g.to_bus, g.kind) for g in bb.groups]
    assert kinds == [
        ("mv", "lv", "transformer"),
        ("lv", "b1", "line"),
        ("b1", "b2", "line"),
        ("b2", "b3", "line"),
    ]
    assert bb.shunts == ()  # the grounding reactor has no phase node


def test_a_bank_is_one_branch(tmp_path: Path) -> None:
    model = prepare_network(network(tmp_path, BANK), root=tmp_path).model
    bb = find_backbone(model)
    assert bb.buses == ("src", "r", "a")
    bank = bb.groups[0]
    by_idx = {e.idx: e.name for e in model.elements}
    names = sorted(by_idx[e] for e in bank.elements)
    assert bank.kind == "transformer"
    assert names == ["Transformer.ra", "Transformer.rb", "Transformer.rc"]
    assert [b for b, _ in bb.shunts] == ["a"]
    assert bb.owner["x"] == "a"


def test_transformer_parameters(tmp_path: Path) -> None:
    model = prepare_network(network(tmp_path, BANK), root=tmp_path).model
    ix, bb = ModelIndex(model), find_backbone(model)
    v_ln = {n.bus: n.kv_base_ln * 1e3 for n in model.nodes}
    p = transformer_params(ix, bb.groups[0], v_ln)
    # a wye bank: no-load positive-sequence ratio mean(t), leakage at the tapped secondary voltage
    t = np.array(TAPS)
    assert p.t == pytest.approx(1 / t.mean(), rel=1e-6)
    assert p.phi_deg == 0.0
    assert p.s_ref_va == pytest.approx(500e3)
    expected = (0.002 + 0.01j) * np.mean(t**2) * (7200 / v_ln["r"]) ** 2
    assert p.z_own == pytest.approx(expected, rel=1e-5)
    # a delta-wye transformer: the vector group, rounded, and the nominal ratio
    model = prepare_network(network(tmp_path / "f", VARIANT_FEEDER), root=tmp_path / "f").model
    v_ln = {n.bus: n.kv_base_ln * 1e3 for n in model.nodes}
    q = transformer_params(ModelIndex(model), find_backbone(model).groups[0], v_ln)
    assert abs(q.phi_deg) == 30.0
    assert q.t == pytest.approx(1.0, abs=1e-5)
    assert q.z_own == pytest.approx(0.01 + 0.04j, rel=1e-4)


# --------------------------------------------------------------------------------- the view


def solved(
    tmp_path: Path, circuit: dict[str, str], n: int = 4, **sections: Any
) -> tuple[PosSeqView, list[tuple[Any, Any]]]:
    runner, cfg = runner_of(tmp_path, circuit, **sections)
    view = view_of(runner, cfg)
    pairs = []
    for s in range(n):
        o = stored(runner.run(s))
        pairs.append((o, view.solve(o)))
    return view, pairs


def datakit_of(view: PosSeqView, pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    """datakit's tables of the view's scenarios, as the export composes them."""
    states = view.state_tables()
    static = ViewNetwork.of(
        states["views/posseq/buses"], states["views/posseq/branches"],
        states["views/posseq/ybus"], view.generators_table(),
    )  # fmt: skip
    done = [o for o, p in pairs if p.status == "converged"]
    state_of = {o.scenario_id: o.state_id for o in done}
    variant_of = {o.scenario_id: o.variant_id for o in done}
    index = {sid: k for k, sid in enumerate(sorted(state_of))}
    results = view.tables(pairs)["views/posseq/bus_results"]
    return compose(results, static, state_of, variant_of, index)


@pytest.mark.parametrize("circuit", [VARIANT_FEEDER, BANK], ids=["variant_feeder", "bank"])
def test_the_view_satisfies_datakits_power_balance(tmp_path: Path, circuit: dict[str, str]) -> None:
    view, pairs = solved(tmp_path, circuit, der={"pv": {"penetration": 1.0}})
    assert [p.status for _, p in pairs] == ["converged"] * len(pairs)
    for name, table in {**view.tables(pairs), **view.state_tables()}.items():
        schema.check_schema(table, schema.POSSEQ_TABLES[name], name)
    tables = datakit_of(view, pairs)
    bus, branch = tables["bus"], tables["branch"]
    mismatch_mw = power_balance(bus, branch)
    assert mismatch_mw.max() < 1e-6 * view.s_base_va / 1e6  # AC; measured <= 1e-8
    n = len(view.backbone.buses)
    assert bus.num_rows == len(pairs) * n
    assert tables["gen"].num_rows == len(pairs)
    # the yprim read back equals datakit's formula from r, x, b, tap
    yff = branch.column("Yff_r").to_numpy() + 1j * branch.column("Yff_i").to_numpy()
    r, x = branch.column("r").to_numpy(), branch.column("x").to_numpy()
    b, tap = branch.column("b").to_numpy(), branch.column("tap").to_numpy()
    np.testing.assert_allclose(yff, (1 / (r + 1j * x) + 0.5j * b) / tap**2, rtol=1e-12)


def test_der_enter_the_net_demand(tmp_path: Path) -> None:
    view, pairs = solved(tmp_path, VARIANT_FEEDER, der={"pv": {"penetration": 1.0}})
    bus = view.tables(pairs)["views/posseq/bus_results"]
    n = len(view.backbone.buses)
    for k, (o, _) in enumerate(pairs):
        rows = slice(k * n, (k + 1) * n)
        der_p = bus.column("x_der_p_mw").to_numpy()[rows]
        assert der_p.sum() == pytest.approx(o.pv_p_w.sum() / 1e6, rel=1e-12)
        assert bus.column("x_der_p_avail_mw").to_numpy()[rows].sum() == pytest.approx(
            o.pv_p_avail_w.sum() / 1e6, rel=1e-12
        )
    gen = datakit_of(view, pairs)["gen"]
    assert gen.column("is_slack_gen").to_pylist() == [1] * len(pairs)  # DER are no generators


def test_the_demand_closes_the_full_models_balance(tmp_path: Path) -> None:
    """Sum of the equivalent demands + losses of the realised elements = source power."""
    runner, cfg = runner_of(tmp_path, VARIANT_FEEDER, der={"pv": {"penetration": 0.5}})
    view = view_of(runner, cfg)
    realised = {e for g in view.backbone.groups for e in g.elements}
    rows = np.isin(runner.plan.element_idx, sorted(realised))
    for s in range(3):
        o = runner.run(s)
        runner.ctx.activate_element("Vsource.source")
        powers = complex_from_interleaved(runner.ctx.api.CktElement.Powers()) * 1e3
        supplied = -powers[:3].sum()
        assert o.v is not None
        assert o.i_pd is not None
        s_rows = (runner.plan.select @ o.v) * np.conj(o.i_pd)
        losses = s_rows[rows].sum()
        demand = view.demand(stored(o))
        assert demand.sum() + losses == pytest.approx(supplied, rel=1e-9)
        # b2's demand: its load h3, the lateral's load h5 and losses (lateral, earthing), less PV
        at_b2 = {"Load.h3", "Load.h5"}
        setpoints = sum(
            complex(p, q)
            for c, p, q in zip(runner.customers, o.load_p_w, o.load_q_var, strict=True)
            if c.element in at_b2
        )
        assert runner.pv is not None
        assert o.pv_p_w is not None
        assert o.pv_q_var is not None
        setpoints -= sum(
            complex(p, q)
            for u, p, q in zip(runner.pv.units, o.pv_p_w, o.pv_q_var, strict=True)
            if u.load in at_b2
        )
        losses_b2 = demand[view.backbone.index()["b2"]].real - setpoints.real
        assert 0 < losses_b2 < 0.05 * abs(setpoints)


def test_the_truth_is_rotated_by_the_vector_group(tmp_path: Path) -> None:
    view, pairs = solved(tmp_path, VARIANT_FEEDER)
    assert list(view.shift_deg) in ([0, 30, 30, 30, 30], [0, -30, -30, -30, -30])
    for _, p in pairs:
        gap = np.degrees(np.abs(np.angle(p.v / p.truth)))
        assert gap.max() < 1.0  # without the rotation: 30 degrees
        assert np.abs(np.abs(p.v) - np.abs(p.truth)).max() < 0.05
        assert p.truth[0] == pytest.approx(abs(p.truth[0]))  # the source at angle 0


def test_a_scenario_does_not_depend_on_the_ones_before(tmp_path: Path) -> None:
    """Bitwise, whatever the view solved before."""
    _, history = solved(tmp_path / "a", VARIANT_FEEDER, n=3, der={"pv": {"penetration": 0.5}})
    runner, cfg = runner_of(tmp_path / "b", VARIANT_FEEDER, der={"pv": {"penetration": 0.5}})
    alone = view_of(runner, cfg).solve(stored(runner.run(2)))
    assert alone.v is not None
    assert np.array_equal(alone.v, history[2][1].v)  # 1.3e-13 apart without the fix


def test_failures_are_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, cfg = runner_of(tmp_path, BANK)
    view = view_of(runner, cfg)
    o = stored(runner.run(0))
    monkeypatch.setattr(posseq, "BALANCE_TOL", -1.0)
    p = view.solve(o)
    assert p.status == "off_setpoint"
    assert p.message is not None
    assert "power mismatch" in p.message
    tables = view.tables([(o, p)])
    assert tables["views/posseq/scenarios"].column("view_status").to_pylist() == ["off_setpoint"]
    assert tables["views/posseq/bus_results"].num_rows == 0
    monkeypatch.undo()
    assert view.solve(o).status == "converged"


def test_mapping_table(tmp_path: Path) -> None:
    runner, cfg = runner_of(tmp_path, VARIANT_FEEDER)
    table = view_of(runner, cfg).mapping_table()
    schema.check_schema(table, schema.POSSEQ_MAPPING, "views/posseq/mapping")
    buses = table.column("backbone_bus").to_pylist()
    rows = dict(zip(buses, table.column("aggregated_buses").to_pylist(), strict=True))
    assert json.loads(rows["b2"]) == ["b4"]
    assert table.column("posseq_bus").to_pylist() == list(range(5))


# ----------------------------------------------------------------------------- datakit's format


def test_columns_are_datakits() -> None:
    for kind, columns in (
        ("bus", datakit.BUS_COLUMNS),
        ("gen", datakit.GEN_COLUMNS),
        ("branch", datakit.BRANCH_COLUMNS),
        ("y_bus", datakit.YBUS_COLUMNS),
    ):
        names = schema.DATAKIT_TABLES[kind].names
        assert names[: len(columns)] == columns, kind
        assert all(c.startswith("x_") for c in names[len(columns) :]), kind


def test_columns_equal_an_installed_datakit() -> None:
    column_names = pytest.importorskip("gridfm_datakit.utils.column_names")
    for attr in ("BUS_COLUMNS", "GEN_COLUMNS", "BRANCH_COLUMNS", "YBUS_COLUMNS"):
        assert getattr(column_names, attr) == getattr(datakit, attr), attr
