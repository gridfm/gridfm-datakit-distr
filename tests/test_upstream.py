"""The MV/LV substation behind the source of the CSIRO feeders (``variants.upstream``)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.source import read_source, scenario_e
from gridfm_datakit_distr.io.reader import DatasetReader
from gridfm_datakit_distr.networks.catalog import ArchivesSource, Catalog, Network, load_catalog
from gridfm_datakit_distr.networks.compile import compile_network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, prepare_network
from gridfm_datakit_distr.scenarios.seeds import network_key
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.solve.pf_batch import VariantRunner, prepare_options
from gridfm_datakit_distr.validate.closed_loop import closed_loop
from gridfm_datakit_distr.variants import upstream as up
from gridfm_datakit_distr.variants.generate import VariantResult, generate_variant
from gridfm_datakit_distr.variants.targets import collect_targets
from tests.circuits import write_circuit
from tests.test_source_unbalance import READBACK_REL, sequences, thevenin

SEED = 5
A = np.exp(2j * np.pi / 3)

# Two feeders of one substation, as the CSIRO four-wire models: an ideal 0.416 kV source with
# its neutral bonded at the source bus, single-phase loads phase to neutral rated 0.23 kV.
_HEAD = """\
Clear
Set DefaultBaseFreq=50
New Circuit.{name} BasekV=0.416 pu=1.00 ISC3=100000 ISC1=100000
New Reactor.Grounding phases=1 bus1=sourcebus.4 bus2=sourcebus.0 R=0.0 X=1E-10
New Linecode.lc4 nphases=4 units=km
~ rmatrix=(0.3 | 0.05 0.3 | 0.05 0.05 0.3 | 0.05 0.05 0.05 0.3)
~ xmatrix=(0.08 | 0.03 0.08 | 0.03 0.03 0.08 | 0.03 0.03 0.03 0.08)
~ cmatrix=(0 | 0 0 | 0 0 0 | 0 0 0 0)
New Line.l1 bus1=sourcebus.1.2.3.4 bus2=b1.1.2.3.4 phases=4 linecode=lc4 length=0.1 units=km
New Line.l2 bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 phases=4 linecode=lc4 length=0.1 units=km
"""
_TAIL = "Set VoltageBases=[0.416]\nCalcVoltageBases\n"
FEEDER = _HEAD.format(name="f1") + """\
New Load.h1 bus1=b1.1.4 phases=1 kV=0.23 kW=2 PF=0.95
New Load.h2 bus1=b2.2.4 phases=1 kV=0.23 kW=3 PF=0.95
New Load.h3 bus1=b2.3.4 phases=1 kV=0.23 kW=1.5 PF=0.95
New Load.h4 bus1=b2.1.4 phases=1 kV=0.23 kW=2.5 PF=0.95
""" + _TAIL  # fmt: skip
SIBLING = _HEAD.format(name="f2") + """\
New Load.s1 bus1=b1.1.4 phases=1 kV=0.23 kW=4 PF=0.9
New Load.s2 bus1=b1.2.4 phases=1 kV=0.23 kW=6 PF=0.9
New Load.s3 bus1=b2.2.4 phases=1 kV=0.23 kW=2 PF=0.9
New Load.s4 bus1=b2.3.4 phases=1 kV=0.23 kW=1 PF=0.9
""" + _TAIL  # fmt: skip
SIBLING_P_W = (4e3, 8e3, 1e3)
TAN_SIBLING = math.tan(math.acos(0.9))
BASE: dict[str, Any] = {
    "dataset_name": "b4",
    "networks": {"include": ["csiro_enwl/n01_f1"]},
    "variants": {"n_variants": 1, "upstream": {"enabled": True}},
    "der": {"pv": {"penetration": 1.0, "kwp": 3.0}},
    "scenarios": {"n_scenarios_per_variant": 6},
    "settings": {"seed": SEED, "num_processes": 1, "residual_sample_rate": 1.0},
}


def catalog(root: Path) -> Catalog:
    """Feeders ``csiro_enwl/n01_f1`` and ``n01_f2`` of one substation (our circuits)."""
    write_circuit(root / "src" / "f1", {"master.dss": FEEDER})
    write_circuit(root / "src" / "f2", {"master.dss": SIBLING})
    source = ArchivesSource(
        kind="archives", id="src", title="test", homepage="-", license="CC0-1.0",
        citation="-", archives={},
    )  # fmt: skip
    networks = {
        f"csiro_enwl/n01_f{k}": Network(id=f"csiro_enwl/n01_f{k}", source="src",
                                        master=f"f{k}/master.dss", neutral="explicit")
        for k in (1, 2)
    }  # fmt: skip
    return Catalog(sources={"src": source}, networks=networks)


def config(**sections: Any) -> GenerationConfig:
    data = copy.deepcopy(BASE)
    for key, value in sections.items():
        data[key] = {**data.get(key, {}), **value}
    return GenerationConfig.model_validate(data)


def site(root: Path) -> tuple[Network, up.Upstream]:
    cat = catalog(root)
    network = cat.network("csiro_enwl/n01_f1")
    return network, up.upstream_of(network, prepare_network(network, root=root), cat, root)


def variant(
    root: Path, cfg: GenerationConfig, level: bool = False
) -> tuple[Network, VariantResult]:
    """Variant 0; with ``level``, its load level drawn from the config."""
    network, substation = site(root)
    options = prepare_options(cfg)
    targets = collect_targets(prepare_network(network, options, root))
    result = generate_variant(
        network, targets, cfg.variants, SEED, 0, options=options, root=root, der=cfg.der.pv,
        upstream=substation if cfg.variants.upstream.enabled else None,
        load_level=cfg.scenarios.profiles.synthetic.variant_scale if level else None,
    )  # fmt: skip
    assert result.status == "ok", result.failures
    return network, result


# ---------------------------------------------------------------------------- the tables


def test_the_rating_table_is_the_derived_one() -> None:
    assert sorted(up.RATING_COUNTS) == [4, 8, 16, 32, 64, 128, 256, 512]
    assert sum(sum(c.values()) for c in up.RATING_COUNTS.values()) == 18133
    assert all(25 <= kva <= 3150 for c in up.RATING_COUNTS.values() for kva in c)
    for customers, lower in ((300, 256), (511, 256), (512, 512), (5000, 512), (4, 4)):
        ratings, p = up.rating_pmf(customers)
        counts = up.RATING_COUNTS[lower]
        np.testing.assert_array_equal(ratings, sorted(counts))
        np.testing.assert_allclose(p, [counts[r] / sum(counts.values()) for r in sorted(counts)])
    with pytest.raises(ValueError, match="starts at 4"):
        up.rating_pmf(3)


def test_a_drawn_rating_follows_the_table() -> None:
    ratings, p = up.rating_pmf(300)
    rng = np.random.default_rng(1)
    draws = np.array([up.draw_rating(300, rng) for _ in range(20000)])
    freq = np.array([(draws == r).mean() for r in ratings])
    assert np.abs(freq - p).max() < 0.012
    assert set(draws) <= set(ratings)


def _pk(kva: float, low: float, high: float, p_low: float, p_high: float) -> float:
    return p_low + (kva - low) / (high - low) * (p_high - p_low)


@pytest.mark.parametrize(
    ("kva", "z", "pk"),
    [
        (25.0, 0.04, 600.0),
        (300.0, 0.04, _pk(300, 250, 315, 2350, 2800)),
        (630.0, 0.04, 4600.0),
        (750.0, 0.05, _pk(750, 630, 800, 4600, 6000)),
        (800.0, 0.05, 6000.0),
        (1000.0, 0.05, 7600.0),
        (1500.0, 0.06, _pk(1500, 1250, 1600, 9500, 12000)),
    ],
)
def test_the_impedance_of_a_rating(kva: float, z: float, pk: float) -> None:
    """IEC 60076-5 Table 1 minimum z_k; resistance from the Tier 2 load losses of Regulation
    548/2014 Table I.1, linear between its rows; reactance the rest of z_k."""
    r, x = up.transformer_impedance(kva)
    assert r == pytest.approx(pk / (kva * 1e3), rel=1e-12)
    assert math.hypot(r, x) == pytest.approx(z, rel=1e-12)
    assert x / r > 1.0


def test_ratings_outside_the_tables_are_refused() -> None:
    for kva in (20.0, 4000.0):
        with pytest.raises(ValueError, match="outside the tables"):
            up.transformer_impedance(kva)


def test_the_source_impedance_adds_the_mv_network_to_z1_only() -> None:
    kv = 0.416
    z1, z0 = up.source_impedance(800.0, kv)
    r, x = up.transformer_impedance(800.0)
    zt = complex(r, x) * (kv * 1e3) ** 2 / 800e3
    assert z0 == pytest.approx(zt, rel=1e-14)
    zmv = z1 - z0
    assert abs(zmv) == pytest.approx((kv * 1e3) ** 2 / up.MV_FAULT_VA, rel=1e-12)
    assert np.angle(zmv) == pytest.approx(np.angle(zt), abs=1e-12)  # the transformer's X/R


def test_the_sequence_admittance() -> None:
    z1, z0 = 0.002 + 0.011j, 0.0015 + 0.010j
    t = np.array([[1, 1, 1], [1, A**2, A], [1, A, A**2]])
    z = t @ np.diag([z0, z1, z1]) @ np.linalg.inv(t)
    np.testing.assert_allclose(up.sequence_admittance(z1, z0), np.linalg.inv(z), rtol=1e-12)


# ------------------------------------------------------------------- scope and refusals


def test_only_csiro_feeders_are_in_scope(tmp_path: Path) -> None:
    assert up.in_scope("csiro_enwl/n13_f1")
    assert not up.in_scope("ieee13")
    assert not up.in_scope("csiro_enwl/n13_f1x")
    networks = [Network(id=i, source="s", master="m") for i in ("ieee13", "csiro_enwl/n01_f1",
                                                                "edp/rural")]  # fmt: skip
    assert up.upstream_issues(networks) == ["edp/rural", "ieee13"]
    cat = catalog(tmp_path)
    other = Network(id="t/x", source="src", master="f1/master.dss")
    with pytest.raises(ValueError, match="csiro_enwl feeders only"):
        up.upstream_of(other, prepare_network(other, root=tmp_path), cat, tmp_path)


def test_generate_refuses_networks_outside_the_scope(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app

    config_path = tmp_path / "c.yaml"
    config_path.write_text(
        "dataset_name: d\nscenarios: {n_scenarios_per_variant: 1}\nsettings: {seed: 1}\n"
        "variants: {upstream: {enabled: true}}\n"
        "networks:\n  include: [ieee13, csiro_enwl/n01_f1, edp/rural]\n"
    )
    result = CliRunner().invoke(app, ["generate", str(config_path), "--dry-run",
                                      "--cache", str(tmp_path / "none")])  # fmt: skip
    assert result.exit_code == 2, result.output
    expected = "applies to csiro_enwl feeders only, not edp/rural, ieee13"
    assert f"variants.upstream.enabled: {expected}" in result.output


def test_a_loadshape_provider_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"variants\.upstream needs"):
        config(scenarios={"profiles": {"provider": "network_loadshapes"}})


# ----------------------------------------------------------------------------- variants


def test_the_substation_is_read_from_the_engine(tmp_path: Path) -> None:
    _, s = site(tmp_path)
    assert s.siblings == ("csiro_enwl/n01_f2",)
    assert s.customers == 8
    assert s.vsource == "Vsource.source"
    assert s.phase_nodes == ("sourcebus.1", "sourcebus.2", "sourcebus.3")
    assert (s.neutral, s.kv, s.kv_ll) == ("4", 0.23, 0.416)
    np.testing.assert_allclose(s.p_w, SIBLING_P_W, rtol=1e-12)
    np.testing.assert_allclose(s.q_var, np.array(SIBLING_P_W) * TAN_SIBLING, rtol=1e-9)
    assert s.loads == tuple(f"Load.upstream_feeders_{p}" for p in "abc")
    assert s.bus1(1) == "sourcebus.2.4"


def test_a_variant_realises_its_substation(tmp_path: Path) -> None:
    """The source gets the drawn rating's impedances (read back), the other feeders three loads
    on the source bus; no node is added; the lumped loads get no PV and are not moved."""
    cfg = config(variants={"phase_reallocation": {"enabled": True, "fraction": 1.0}})
    network, result = variant(tmp_path / "on", cfg)
    p = result.perturbation
    assert p is not None
    assert p.upstream is not None
    assert p.upstream.rating_kva in up.RATING_COUNTS[8]
    z1, z0 = up.source_impedance(p.upstream.rating_kva, 0.416)
    assert (p.upstream.z1_ohm, p.upstream.z0_ohm) == (z1, z0)
    assert result.script[-4].startswith("Edit Vsource.source Z1=")
    prepared = prepare_network(network, prepare_options(cfg), tmp_path / "on",
                               edits=result.script)  # fmt: skip
    y = up.sequence_admittance(z1, z0)
    y_s = read_source(prepared.ctx, prepared.model).y_s
    assert np.abs(y_s - y).max() <= 1e-12 * np.abs(y).max()
    base = prepare_network(network, root=tmp_path / "on")
    assert [(n.opendss_node, n.conductor) for n in prepared.model.nodes] == [
        (n.opendss_node, n.conductor) for n in base.model.nodes
    ]
    lumped = set(p.upstream.loads)
    assert lumped == {f"Load.upstream_feeders_{x}" for x in "abc"}
    assert lumped.isdisjoint(p.pv_kwp)
    assert lumped.isdisjoint(p.phases)
    assert len(p.pv_kwp) == len(p.phases) == 4  # every customer
    record = json.loads(p.to_json())["upstream"]
    assert record["rating_kva"] == p.upstream.rating_kva
    assert record["z0_ohm"] == [z0.real, z0.imag]
    assert record["loads"]["Load.upstream_feeders_b"][0] == SIBLING_P_W[1]
    off = variant(tmp_path / "off", config(variants={"upstream": {"enabled": False}}))[1]
    assert off.perturbation is not None
    assert "upstream" not in json.loads(off.perturbation.to_json())
    assert not any("Vsource" in c or "upstream" in c for c in off.script)


def test_the_lumped_loads_follow_the_global_factor(tmp_path: Path) -> None:
    """Each lumped load draws g times its phase's nominal sum; the customers draw what they draw
    without the substation (same streams)."""
    cfg = config()
    network, result = variant(tmp_path / "on", cfg)
    r = VariantRunner(network, result, cfg, 0, root=tmp_path / "on")
    cfg_off = config(variants={"upstream": {"enabled": False}})
    network_off, result_off = variant(tmp_path / "off", cfg_off)
    r_off = VariantRunner(network_off, result_off, cfg_off, 0, root=tmp_path / "off")
    assert [c.element for c in r.customers] == [c.element for c in r_off.customers]
    assert r.load_elements == (*(c.element for c in r.customers),
                               *(f"Load.upstream_feeders_{x}" for x in "abc"))  # fmt: skip
    assert r.der_penetration == r_off.der_penetration == 1.0
    net = network_key(network.id)
    for s in range(4):
        o, o_off = r.run(s), r_off.run(s)
        assert o.pf_status == o_off.pf_status == "converged", o.message  # setpoints met
        g = r.loads.draw(SEED, net, 0, s).global_scale
        assert g is not None
        np.testing.assert_array_equal(o.load_p_w[-3:], g * np.array(SIBLING_P_W))
        np.testing.assert_allclose(o.load_q_var[-3:], g * np.array(SIBLING_P_W) * TAN_SIBLING,
                                   rtol=1e-9)  # fmt: skip
        np.testing.assert_array_equal(o.load_p_w[:-3], o_off.load_p_w)
        np.testing.assert_array_equal(o.load_q_var[:-3], o_off.load_q_var)
        assert o.v is not None
        assert o_off.v is not None
        bus = [n.idx for n in r.model.nodes if n.bus == "sourcebus" and n.conductor != "n"]
        assert np.abs(o.v[bus]).max() < np.abs(o_off.v[bus]).min()  # the substation's drop


def test_the_lumped_loads_follow_the_load_level(tmp_path: Path) -> None:
    """With a load level ``L`` each lumped load draws ``L · g`` times its nominal sum."""
    cfg = config()
    network, result = variant(tmp_path, cfg, level=True)
    assert result.perturbation is not None
    level = result.perturbation.variant_scale
    assert level != 1.0
    r = VariantRunner(network, result, cfg, 0, root=tmp_path)
    for s in range(2):
        o = r.run(s)
        g = r.loads.draw(SEED, network_key(network.id), 0, s).global_scale
        assert g is not None
        np.testing.assert_allclose(o.load_p_w[-3:], level * g * np.array(SIBLING_P_W),
                                   rtol=1e-15)  # fmt: skip


def test_the_negative_sequence_source_sees_the_substation(tmp_path: Path) -> None:
    """With Z0 != Z1 the source admittance stays circulant: B2's current source realises the drawn
    ideal voltages behind the substation's impedance."""
    cfg = config(scenarios={"source_unbalance": {"distribution": "uniform", "low": 0.005,
                                                 "high": 0.02}})  # fmt: skip
    network, result = variant(tmp_path, cfg)
    r = VariantRunner(network, result, cfg, 0, root=tmp_path)
    assert r.unbalance is not None
    source = read_source(r.ctx, r.model)
    for s in range(3):
        o = r.run(s)
        assert o.pf_status == "converged", o.message
        e = scenario_e(source.e_v, o.source_v_pu, o.source_u2, o.source_u2_angle_rad)
        got = thevenin(r.ctx, r.model, source)
        assert np.abs(got - e).max() <= READBACK_REL * np.abs(e).max()
        _, pos, neg = sequences(got)
        assert abs(neg / pos) == pytest.approx(o.source_u2, abs=READBACK_REL)


# ------------------------------------------------------- the substation against a real Dyn

MV_BUS = "mvbus"
# Measured max |dV| / V_base: <= 3.6e-8 on n13_f1 (4452 nodes) and on the unit feeder, 500-1500 kVA,
# loads x1 and x3 (2.2e-8 at 500 kVA, 2.9e-8 at 1000): a margin of 2.8. The floor is the same at
# every load (it is there at no load): the engine's transformer, not the formula.
DYN_TOL_PU = 1e-7


def dyn_transformer(site: up.Upstream, kva: float) -> list[str]:
    """The reference: the source moved to an MV bus, behind a real Dyn transformer of the rating's
    impedance (``transformer_impedance``) with the MV network's fault level (angle 30: Dyn11).
    Independent of the Thevenin formula, built from the engine's own elements."""
    r, x = up.transformer_impedance(kva)
    bus = site.phase_nodes[0].rsplit(".", 1)[0]
    mva = up.MV_FAULT_VA / 1e6
    return [
        f"Edit {site.vsource} bus1={MV_BUS} basekV=11 angle=30 MVAsc3={mva!r} MVAsc1={mva!r} "
        f"X1R1={x / r!r} X0R0={x / r!r}",
        f"New Transformer.sub phases=3 windings=2 buses=[{MV_BUS}.1.2.3 {bus}.1.2.3.4] "
        f"conns=[delta wye] kVs=[11 {site.kv_ll!r}] kVAs=[{kva!r} {kva!r}] "
        f"XHL={100 * x!r} %Rs=[{50 * r!r} {50 * r!r}] %imag=0 %noloadloss=0",
    ]  # fmt: skip


def node_voltages(
    network: Network,
    root: Path | None,
    site: up.Upstream,
    edits: list[str],
    scales: Sequence[float],
) -> list[dict[str, complex]]:
    """Voltages to ground (V) at every node of ``network`` with ``edits`` and the lumped sibling
    loads, one solve per scale: each load at ``scale`` times its nominal power (the lumped ones
    at ``scale`` times the sums of their phase, as a scenario's global load factor does)."""
    ctx = DssContext.new()
    compile_network(ctx, network, root)
    for command in [*edits, *up.load_commands(site)]:
        ctx.command(command)
    ctx.command("Set Tolerance=1e-12 MaxIterations=100")
    lumped = {load.split(".", 1)[1]: k for k, load in enumerate(site.loads)}
    nominal = {}
    for name in ctx.api.Loads.AllNames():
        ctx.api.Loads.Name(name)
        k = lumped.get(name)
        nominal[name] = (
            (site.p_w[k] / 1e3, site.q_var[k] / 1e3)
            if k is not None
            else (ctx.api.Loads.kW(), ctx.api.Loads.kvar())
        )
    out = []
    for scale in scales:
        for name, (kw, kvar) in nominal.items():
            ctx.api.Loads.Name(name)
            ctx.api.Loads.kW(scale * kw)
            ctx.api.Loads.kvar(scale * kvar)
        ctx.command("Set Mode=Snap")
        ctx.command("Solve")
        assert ctx.api.Solution.Converged()
        out.append(dict(zip(ctx.ynode_order(), ctx.ynode_voltages(), strict=True)))
    return out


def dyn_deviations(
    network: Network,
    root: Path | None,
    site: up.Upstream,
    base: PreparedNetwork,
    ratings: Sequence[float],
    scales: Sequence[float],
) -> dict[tuple[float, float], dict[str, float]]:
    """Per (rating, scale): ``nodes`` compared, and max |dV| / V_base over them of the
    source with the substation's sequence impedances (``thevenin``), of the ideal source
    (``stiff``) and of the same with ``Z0 = Z1`` (``z0_as_z1``), each against the real Dyn
    transformer. The last two are the controls: the comparison must be able to tell them apart.
    The Thevenin model's node set is the source network's, the compared nodes are all of them,
    neutrals included."""
    v_base = site.kv_ll * 1e3 / math.sqrt(3)
    stiff = node_voltages(network, root, site, [], scales)
    source_nodes = {n.opendss_node for n in base.model.nodes}
    out = {}
    for kva in ratings:
        z1, z0 = up.source_impedance(kva, site.kv_ll)
        script = up.source_edit(site.vsource, z1, z0)
        edited = prepare_network(network, root=root, edits=[*script, *up.load_commands(site)])
        assert [n.opendss_node for n in edited.model.nodes] == [
            n.opendss_node for n in base.model.nodes
        ]  # no node added
        models = {
            "thevenin": node_voltages(network, root, site, script, scales),
            "z0_as_z1": node_voltages(network, root, site, up.source_edit(site.vsource, z1, z1),
                                      scales),
            "stiff": stiff,
        }  # fmt: skip
        real = node_voltages(network, root, site, dyn_transformer(site, kva), scales)
        for j, scale in enumerate(scales):
            nodes = [n for n in real[j] if not n.startswith(f"{MV_BUS}.")]
            assert set(nodes) == set(models["thevenin"][j]) == set(stiff[j]) == source_nodes
            assert any(n.endswith(f".{site.neutral}") for n in nodes)
            out[kva, scale] = {"nodes": len(nodes)} | {
                key: max(abs(v[j][n] - real[j][n]) for n in nodes) / v_base
                for key, v in models.items()
            }
    return out


def assert_matches_dyn(devs: dict[tuple[float, float], dict[str, float]], nodes: int) -> None:
    for (kva, scale), d in devs.items():
        where = f"{kva:g} kVA, loads x{scale:g}: {d}"
        assert d["nodes"] == nodes, where
        assert d["thevenin"] <= DYN_TOL_PU, where
        # not vacuous: a source without the substation, or without its Z0, is far outside
        assert d["stiff"] > 100 * DYN_TOL_PU, where
        assert d["z0_as_z1"] > 100 * DYN_TOL_PU, where


@pytest.mark.network_data
def test_the_substation_is_a_dyn_transformer_at_every_node_of_a_catalog_feeder(
    network_master: Callable[[str], Path],
) -> None:
    """The source with the substation's sequence impedances and the lumped sibling feeders give
    the voltages of a real Dyn transformer fed by the MV network, at all 4452 nodes (neutrals
    included) of csiro_enwl/n13_f1, and add no node."""
    network_master("csiro_enwl/n13_f1")  # skips without the cache
    cat = load_catalog()
    network = cat.network("csiro_enwl/n13_f1")
    base = prepare_network(network)
    devs = dyn_deviations(network, None, up.upstream_of(network, base, cat), base,
                          (500.0, 1000.0, 1500.0), (1.0, 3.0))  # fmt: skip
    assert len(base.model.nodes) == 4452
    assert_matches_dyn(devs, nodes=4452)


def test_the_substation_is_a_dyn_transformer_on_a_four_wire_feeder(tmp_path: Path) -> None:
    """The same comparison on the in-repo four-wire feeder (no network data), over the ratings'
    three impedance classes (4, 5 and 6 %) and the smallest."""
    network, substation = site(tmp_path)
    base = prepare_network(network, root=tmp_path)
    devs = dyn_deviations(network, tmp_path, substation, base, (25.0, 500.0, 1000.0, 1500.0),
                          (1.0, 3.0))  # fmt: skip
    assert_matches_dyn(devs, nodes=len(base.model.nodes))


# ------------------------------------------------------------------------------ dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("casadi")
    root = tmp_path_factory.mktemp("b4")
    cfg = config(views={"kron": True}, labels={"opf": {"enabled": True}})
    out = root / "out"
    generate(cfg, out, catalog=catalog(root), root=root)
    return out


def test_the_dataset_carries_the_substation(dataset: Path) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text())
    [entry] = manifest["networks"]
    assert entry["upstream"]["siblings"] == ["csiro_enwl/n01_f2"]
    assert entry["upstream"]["customers"] == 8
    [variant_row] = pq.read_table(dataset / "variants/variants").to_pylist()
    record = json.loads(variant_row["perturbation"])["upstream"]
    y = up.sequence_admittance(complex(*record["z1_ohm"]), complex(*record["z0_ohm"]))
    stored = np.zeros((3, 3), dtype=np.complex128)
    for row in pq.read_table(dataset / "variants/source_yprim").to_pylist():
        stored[row["row"], row["col"]] = complex(row["g"], row["b"])
    assert np.abs(stored - y).max() <= 1e-12 * np.abs(y).max()
    elements = {r["element_idx"]: r["name"] for r in
                pq.read_table(dataset / "variants/elements").to_pylist()}  # fmt: skip
    loads = pq.read_table(dataset / "scenarios/loads").to_pylist()
    lumped = [r for r in loads if elements[r["element_idx"]] in record["loads"]]
    assert len(lumped) == 3 * 6
    meta = pq.read_table(dataset / "scenarios/scenario_meta").to_pylist()
    assert {m["pf_status"] for m in meta} == {"converged"}


def test_the_stored_tables_reproduce_the_voltages(dataset: Path) -> None:
    reader = DatasetReader(dataset)
    for sc in reader.scenarios(reader.meta.column("scenario_id").to_pylist()):
        assert closed_loop(sc).max_dv_pu <= 1e-6


def test_the_opf_is_verified_with_the_substation(dataset: Path) -> None:
    """The lumped loads are fixed loads of the OPF; its dispatch verifies in OpenDSS."""
    opf = pq.read_table(dataset / "opf/opf_meta").to_pylist()
    solved = [r for r in opf if r["opf_status"] == "Solve_Succeeded"]
    assert len(solved) >= len(opf) // 2
    assert all(r["verified"] and r["verify_max_dv_pu"] <= 1e-6 for r in solved)
    kron = pq.read_table(dataset / "views/kron/node_results")
    assert kron.num_rows > 0
