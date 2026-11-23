"""Tests for the Kron view."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.variants.generate import generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from gridfm_datakit_distr.views import kron
from gridfm_datakit_distr.views.kron import KronError, KronView, kron_script
from tests.circuits import FOUR_WIRE, LATERAL, SPLIT_PHASE, VARIANT_FEEDER, write_circuit

CONFIG: dict[str, Any] = {
    "dataset_name": "t",
    "networks": {"include": ["x"]},
    "scenarios": {"n_scenarios_per_variant": 4},
    "settings": {"seed": 5},
}


def network(tmp_path: Path, circuit: dict[str, str]) -> Network:
    write_circuit(tmp_path / "src", circuit)
    return Network(id="t/net", source="src", master="master.dss")


def test_the_reduced_line_is_the_schur_complement(tmp_path: Path) -> None:
    net = network(tmp_path, FOUR_WIRE)
    full = prepare_network(net, root=tmp_path)
    script = kron_script(full.ctx, full.model)
    assert script is not None
    assert "Disable Reactor.ground" in script  # the neutral's bond to the source
    assert "Edit Load.house bus1=b1.1" in script
    view = prepare_network(net, root=tmp_path, variant_id="t/net/v0000", edits=script)
    assert view.solve.converged
    assert not any(n.conductor == "n" for n in view.model.nodes)
    # linecode lc4 (ohm/km) over 0.2 km, neutral eliminated
    r = np.array(
        [[0.5, 0.1, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1], [0.1, 0.1, 0.5, 0.1], [0.1, 0.1, 0.1, 0.6]]
    )
    x = np.array(
        [[0.3, 0.1, 0.1, 0.1], [0.1, 0.3, 0.1, 0.1], [0.1, 0.1, 0.3, 0.1], [0.1, 0.1, 0.1, 0.35]]
    )
    z = (r + 1j * x) * 0.2
    z_red = z[:3, :3] - np.outer(z[:3, 3], z[3, :3]) / z[3, 3]
    line = next(e for e in view.model.elements if e.name == "Line.l1_kron")
    assert line.yprim is not None
    np.testing.assert_allclose(np.linalg.inv(-line.yprim[:3, 3:]), z_red, rtol=1e-12)


def test_a_centre_tap_winding_keeps_its_polarity(tmp_path: Path) -> None:
    """Winding ``x.3.2`` lists the centre (the neutral) first: the view grounds it in place."""
    net = network(tmp_path, SPLIT_PHASE)
    full = prepare_network(net, root=tmp_path)
    script = kron_script(full.ctx, full.model)
    assert script is not None
    assert "Edit Transformer.ct wdg=3 bus=x.0.2 Rneut=-1" in script
    view = prepare_network(net, root=tmp_path, variant_id="t/net/v0000", edits=script)
    assert view.solve.converged

    def v_240(p: Any) -> complex:
        v = dict(zip(p.ctx.ynode_order(), p.ctx.ynode_voltages(), strict=True))
        return complex(v["h.1"] - v["h.2"])

    assert abs(v_240(full)) > 230
    assert v_240(view) == pytest.approx(v_240(full), rel=1e-2)


def test_a_kron_native_network_has_the_identity_view(tmp_path: Path) -> None:
    full = prepare_network(network(tmp_path, LATERAL), root=tmp_path)
    assert kron_script(full.ctx, full.model) is None


def runner_and_variant(
    tmp_path: Path, circuit: dict[str, str], **sections: Any
) -> tuple[VariantRunner, Any, Network]:
    data = copy.deepcopy(CONFIG)
    data.update(sections)
    cfg = GenerationConfig.model_validate(data)
    net = network(tmp_path, circuit)
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(net, options, tmp_path))
    result = generate_variant(
        net, targets, cfg.variants, 5, 0, options=options, root=tmp_path, der=cfg.der.pv,
        storage=cfg.der.storage, dg_kw=5.0 if cfg.der.dg.enabled else None,
    )  # fmt: skip
    return VariantRunner(net, result, cfg, 0, root=tmp_path), result, net


def test_the_view_holds_the_storage_and_dg_setpoints(tmp_path: Path) -> None:
    der = {"pv": {"penetration": 1.0}, "storage": {"share": 1.0}, "dg": {"enabled": True}}
    runner, result, net = runner_and_variant(
        tmp_path, VARIANT_FEEDER, der=der, controls={"mode": "as_is"}
    )
    assert runner.storage is not None
    assert runner.dg is not None
    view = KronView(runner, net, result, root=tmp_path)
    o = runner.run(2)
    assert o.pf_status == "converged", o.message
    assert o.bess_p_w is not None
    assert o.dg_p_w is not None
    assert np.abs(o.bess_p_w).max() > 100.0
    assert o.dg_p_w.max() > 100.0
    k = view.solve(o, runner.registry.states.get(o.state_id or ""))
    assert k.status == "converged", k.message
    api = view.ctx.api
    units = runner.storage.units + runner.dg.units
    for unit, p_w in zip(units, np.r_[o.bess_p_w, o.dg_p_w], strict=True):
        api.Circuit.SetActiveElement(unit.generator)
        delivered = -float(np.asarray(api.CktElement.Powers())[0::2].sum()) * 1e3
        assert delivered == pytest.approx(p_w, rel=1e-6, abs=1e-6)


def test_scenarios_on_the_view(tmp_path: Path) -> None:
    runner, result, net = runner_and_variant(
        tmp_path, VARIANT_FEEDER, der={"pv": {"penetration": 1.0}}, controls={"mode": "as_is"}
    )
    view = KronView(runner, net, result, root=tmp_path)
    assert not view.identity
    pairs = []
    for s in range(4):
        o = runner.run(s)
        assert o.pf_status == "converged", o.message
        k = view.solve(o, runner.registry.states.get(o.state_id or ""))
        assert k.status == "converged", k.message
        assert k.v is not None
        pairs.append((o, k))
    nodes = view.nodes_table()
    schema.check_schema(nodes, schema.KRON_NODES, "views/kron/nodes")
    assert "n" not in nodes.column("conductor").to_pylist()
    results = view.results_table(pairs)
    schema.check_schema(results, schema.KRON_NODE_RESULTS, "views/kron/node_results")
    assert results.num_rows == 4 * nodes.num_rows
    assert "truth_v_re" not in results.column_names  # the truth is node_results at node_idx_full
    truth = pairs[0][0].v[nodes.column("node_idx_full").to_numpy()]  # type: ignore[index]
    # the view differs from the four-wire truth only by the neutral's voltage rise
    gap = np.abs(pairs[0][1].v - truth).max() / 230.94  # type: ignore[operator]
    assert 0 < gap < 0.05


def test_the_meter_trap_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = {"master.dss": FOUR_WIRE["master.dss"].replace(
        "Set VoltageBases", "New EnergyMeter.m element=Line.l1 terminal=1\\nSet VoltageBases"
    ).replace("\\n", "\n")}  # fmt: skip
    runner, result, net = runner_and_variant(tmp_path, circuit)
    real = kron.kron_script

    def without_meter_fix(ctx: Any, model: Any) -> list[str] | None:
        script = real(ctx, model)
        return [c for c in script or [] if not c.startswith("Disable EnergyMeter.")]

    monkeypatch.setattr(kron, "kron_script", without_meter_fix)
    with pytest.raises(KronError, match="fails its base case"):
        KronView(runner, net, result, root=tmp_path)
    monkeypatch.undo()
    assert not KronView(runner, net, result, root=tmp_path).identity  # with the fix it solves
