"""Extraction, residual (C2), solver Y (C3), coverage (C4), phases (C5), neutral (C6), bases,
load setpoints, semantic layer, passports, golden files (C7)."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.extract.extract_semantic import TO_METRE, extract_semantic
from gridfm_datakit_distr.extract.injections import injections, pc_source_admittance
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.ids import base_variant_id
from gridfm_datakit_distr.model.semantic import SemanticModel
from gridfm_datakit_distr.model.units import SQRT3
from gridfm_datakit_distr.networks.catalog import Catalog, load_catalog
from gridfm_datakit_distr.networks.compile import compile_network
from gridfm_datakit_distr.networks.passport import build_passport
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.normalize.bases import check_bases, fix_bases, s_base_va
from gridfm_datakit_distr.normalize.coverage import coverage_report
from gridfm_datakit_distr.normalize.loads import normalize_loads
from gridfm_datakit_distr.normalize.neutral import check_neutral, classify_neutral
from gridfm_datakit_distr.normalize.phases import check_phases, phase_consistency
from gridfm_datakit_distr.validate.consistency import compare_admittance
from gridfm_datakit_distr.validate.golden import compare_golden, load_golden
from gridfm_datakit_distr.validate.residual import residual
from gridfm_datakit_distr.validate.setpoints import setpoint_report

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]

NETWORKS = [
    "ieee13",
    "ieee34",
    "ieee37",
    "ieee123",
    "ieee_eulv",
    "ieee8500",  # split-phase secondaries, 1177 centre-tap transformers
    "csiro_enwl/n14_f6",  # small CSIRO feeder
    "csiro_enwl/n02_f5",
    "csiro_enwl/n21_f3",  # the median CSIRO feeder
    "csiro_enwl/n17_f7",  # 90th percentile, 9760 nodes
    "edp/industrial",
    "edp/rural",
    pytest.param("edp/urban", marks=pytest.mark.slow),  # 25775 nodes, 35297 loads
]


@functools.cache
def _catalog() -> Catalog:
    return load_catalog()


def compiled(network_master: Callable[[str], Path], network_id: str) -> DssContext:
    """The network compiled with its catalog recipe; skipped if not fetched."""
    network_master(network_id)
    ctx = DssContext.new()
    compile_network(ctx, _catalog().network(network_id))
    return ctx


def solved(
    network_master: Callable[[str], Path], network_id: str
) -> tuple[DssContext, ElectricalModel]:
    ctx = compiled(network_master, network_id)
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    assert ctx.solve().converged
    return ctx, extract_electrical(ctx, network_id, base_variant_id(network_id))


@pytest.mark.parametrize("network_id", NETWORKS)
def test_catalog_network_extracts(network_master: Callable[[str], Path], network_id: str) -> None:
    ctx, model = solved(network_master, network_id)
    assert [n.opendss_node for n in model.nodes] == ctx.ynode_order()
    assert model.source_nodes().size == 3
    assert all(e.yprim is not None for e in model.pd_elements())
    used = {n for e in model.elements for n in e.local_nodes() if n >= 0}
    assert used == set(range(model.n_nodes))  # every node belongs to some element


@pytest.mark.parametrize("network_id", NETWORKS)
def test_residual_c2(network_master: Callable[[str], Path], network_id: str) -> None:
    ctx, model = solved(network_master, network_id)
    report = residual(model, ctx.ynode_voltages(), injections(ctx, model), tolerance=1e-10)
    assert report.passes(), report  # C2
    assert report.rel <= 1e-6, report  # the plan's original criterion still holds


@pytest.mark.parametrize("network_id", NETWORKS)
def test_solver_admittance_c3(network_master: Callable[[str], Path], network_id: str) -> None:
    ctx, model = solved(network_master, network_id)
    report = compare_admittance(
        model, model.ybus(), ctx.system_y(), pc_source_admittance(ctx, model)
    )
    assert report.passes(1e-9), report


@pytest.mark.parametrize("network_id", NETWORKS)
def test_element_coverage_c4(network_master: Callable[[str], Path], network_id: str) -> None:
    ctx = compiled(network_master, network_id)
    report = coverage_report(ctx)
    assert report.passes(), report.unsupported_enabled


@pytest.mark.parametrize("network_id", NETWORKS)
def test_neutral_classification_c6(network_master: Callable[[str], Path], network_id: str) -> None:
    _, model = solved(network_master, network_id)
    expected = _catalog().network(network_id).neutral
    check_neutral(classify_neutral(model), expected)


@pytest.mark.parametrize("network_id", NETWORKS)
def test_phase_consistency_c5(network_master: Callable[[str], Path], network_id: str) -> None:
    _, model = solved(network_master, network_id)
    check_phases(phase_consistency(model), strict=True)


@pytest.mark.parametrize("network_id", NETWORKS)
def test_voltage_bases(network_master: Callable[[str], Path], network_id: str) -> None:
    ctx, model = solved(network_master, network_id)
    report = check_bases(ctx, model)
    if network_id.startswith("csiro_enwl/"):  # no VoltageBases: 0.48 kV defaults
        assert not report.ambiguous
        assert not report.conflicts
        assert set(report.rebase) == {n.bus for n in model.nodes}
        fix_bases(ctx, report)
        ctx.solve()
        model = extract_electrical(ctx, network_id, base_variant_id(network_id))
        report = check_bases(ctx, model)
    assert report.passes(), (list(report.rebase)[:3], report.ambiguous[:3], report.conflicts)


def test_ieee13_documented_bases(network_master: Callable[[str], Path]) -> None:
    _, model = solved(network_master, "ieee13")
    kv_ll = {n.bus: round(n.kv_base_ln * SQRT3, 3) for n in model.nodes}
    assert (kv_ll["sourcebus"], kv_ll["650"], kv_ll["671"], kv_ll["634"]) == (
        115.0,
        4.16,
        4.16,
        0.48,
    )


@pytest.mark.parametrize("network_id", NETWORKS)
def test_load_setpoints(network_master: Callable[[str], Path], network_id: str) -> None:
    ctx = compiled(network_master, network_id)
    normalize_loads(ctx)
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    assert ctx.solve().converged
    report = setpoint_report(ctx)
    assert report.passes(), (report.off[:5], report.worst, report.max_rel)


@pytest.mark.parametrize("network_id", NETWORKS)
def test_semantic_layer(network_master: Callable[[str], Path], network_id: str) -> None:
    """The semantic JSON validates; linecode lines carry their linecode matrices."""
    ctx = compiled(network_master, network_id)
    normalize_loads(ctx)
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    assert ctx.solve().converged
    model = extract_electrical(ctx, network_id, base_variant_id(network_id))
    sem = extract_semantic(ctx, model, s_base_va=s_base_va(ctx, model)[0])
    assert SemanticModel.model_validate_json(sem.model_dump_json()) == sem
    api = ctx.api
    checked, worst = 0, 0.0
    for name, line in sem.line.items():
        if not line.source_id.startswith("line."):
            continue  # a series reactor or capacitor stored as a lumped line
        api.Lines.Name(name)
        code, line_unit = api.Lines.LineCode(), int(api.Lines.Units())
        if not code:
            continue
        api.LineCodes.Name(code)
        unit = int(api.LineCodes.Units()) or line_unit
        per = TO_METRE[unit] if line.length_unit == "m" else 1.0
        n = len(line.rs)
        r = np.reshape(api.LineCodes.Rmatrix(), (n, n)) / per
        x = np.reshape(api.LineCodes.Xmatrix(), (n, n)) / per
        for ours, theirs in ((np.array(line.rs), r), (np.array(line.xs), x)):
            worst = max(worst, float(np.abs(ours - theirs).max() / np.abs(theirs).max()))
        checked += 1
    assert checked > 0
    assert worst <= 1e-9, worst


@pytest.mark.parametrize("network_id", NETWORKS)
def test_passport(network_master: Callable[[str], Path], network_id: str) -> None:
    """Passports are written for every network and report passing checks."""
    network_master(network_id)
    passport = build_passport(prepare_network(_catalog().network(network_id)), _catalog())
    required = {"network_id", "source_url", "license", "citation", "checksums",
                "original_format", "converter", "normalisation", "neutral_representation",
                "slack_definition", "known_differences", "excluded_elements"}  # fmt: skip
    assert required <= set(passport)
    assert json.loads(json.dumps(passport)) == passport
    assert all(passport["checks"].values()), passport["checks"]


GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.mark.parametrize("network_id", NETWORKS)
def test_golden_c7(network_master: Callable[[str], Path], network_id: str) -> None:
    """Structure hash and base-case voltages equal the committed golden file."""
    network_master(network_id)
    golden = load_golden(GOLDEN_DIR, network_id)
    if golden is None:
        pytest.fail("no golden file: run integrationtests/golden/regenerate.py")
    prepared = prepare_network(_catalog().network(network_id))
    result = compare_golden(prepared.model, prepared.ctx.ynode_voltages(), golden)
    assert result.passed, result.detail
