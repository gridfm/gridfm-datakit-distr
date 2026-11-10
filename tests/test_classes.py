"""Every supported element class on inline circuits, and the coverage report."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.classes import PC_CLASSES, PD_CLASSES, SOURCE_CLASSES, category
from gridfm_datakit_distr.extract.extract_electrical import (
    UnsupportedElementError,
    extract_electrical,
)
from gridfm_datakit_distr.extract.injections import injections, pc_source_admittance
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.normalize.coverage import coverage_report
from gridfm_datakit_distr.validate.consistency import compare_admittance
from gridfm_datakit_distr.validate.residual import residual
from tests.circuits import ALL_CLASSES, CONTROLLED, TWO_BUS, write_circuit


@pytest.fixture(scope="module")
def all_classes(tmp_path_factory: pytest.TempPathFactory) -> tuple[DssContext, ElectricalModel]:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path_factory.mktemp("all"), ALL_CLASSES))
    ctx.set_options(tolerance=1e-10, max_iterations=100)
    assert ctx.solve().converged
    return ctx, extract_electrical(ctx, "cls", "cls/base")


def element(model: ElectricalModel, name: str) -> Element:
    (found,) = [e for e in model.elements if e.name.lower() == name.lower()]
    return found


def test_every_supported_class_is_present(all_classes: tuple[DssContext, ElectricalModel]) -> None:
    _, model = all_classes
    assert {e.element_class for e in model.elements} == PD_CLASSES | PC_CLASSES | SOURCE_CLASSES


def test_kcl_and_solver_y_hold_for_every_class(
    all_classes: tuple[DssContext, ElectricalModel],
) -> None:
    ctx, model = all_classes
    report = residual(model, ctx.ynode_voltages(), injections(ctx, model), tolerance=1e-10)
    assert report.passes(), report
    assert report.rel <= 1e-8, report
    assert report.backward < 1e-10, report
    c3 = compare_admittance(model, model.ybus(), ctx.system_y(), pc_source_admittance(ctx, model))
    assert c3.passes(1e-12), c3


def test_class_specific_terminal_maps(all_classes: tuple[DssContext, ElectricalModel]) -> None:
    _, model = all_classes
    idx = {n.opendss_node: n.idx for n in model.nodes}
    geo = element(model, "Line.geo")  # 4 wires, Kron-reduced to 3 by the geometry
    assert (geo.n_terminals, geo.n_conductors) == (2, 3)
    assert element(model, "Transformer.t3w").n_terminals == 3
    centre_tap = element(model, "Transformer.ct")  # NodeOrder [1, 0, 1, 0, 0, 2]
    assert centre_tap.terminals == [
        [idx["m.1"], GROUND],
        [idx["ct.1"], GROUND],
        [GROUND, idx["ct.2"]],
    ]
    bank = [element(model, f"Transformer.b{k}") for k in "abc"]
    assert [b.terminals[1][0] for b in bank] == [idx["bk.1"], idx["bk.2"], idx["bk.3"]]
    shunt = element(model, "Reactor.shunt")
    assert shunt.terminals[1] == [GROUND, GROUND, GROUND]
    assert GROUND not in element(model, "Capacitor.cdelta").terminals[0]
    assert element(model, "Load.ldelta").terminals == [[idx["t3b.1"], idx["t3b.2"], idx["t3b.3"]]]
    assert element(model, "Load.lct").terminals == [[idx["ct.1"], idx["ct.2"]]]
    for name in ("Generator.g", "PVSystem.pv", "Storage.st"):
        der = element(model, name)
        assert not der.is_pd
        assert der.yprim is None
        assert der.terminals[0][-1] == GROUND  # wye with the neutral to ground


def test_coverage_report(all_classes: tuple[DssContext, ElectricalModel], tmp_path: Path) -> None:
    ctx, _ = all_classes
    report = coverage_report(ctx)
    counts = {c.element_class: (c.category, c.enabled, c.disabled) for c in report.classes}
    assert counts["transformer"] == ("pd", 5, 0)
    assert counts["load"] == ("pc", 4, 0)
    assert counts["vsource"] == ("source", 1, 0)
    assert report.passes()

    ctrl = DssContext.new()
    ctrl.load(write_circuit(tmp_path / "ctrl", CONTROLLED))
    counts = {
        c.element_class: (c.category, c.enabled, c.disabled) for c in coverage_report(ctrl).classes
    }
    assert counts["line"] == ("pd", 2, 1)  # Line.off is disabled
    assert (
        counts["regcontrol"][0] == counts["capcontrol"][0] == counts["energymeter"][0] == "control"
    )


@pytest.mark.parametrize(
    ("extra", "offender"),
    [
        ("New Isource.is bus1=load.1.2.3 phases=3 amps=2", "Isource.is"),
        ("New Fault.f1 bus1=load.1 phases=1 r=1", "Fault.f1"),
    ],
)
def test_unsupported_classes(tmp_path: Path, extra: str, offender: str) -> None:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, TWO_BUS), extra_commands=(extra,))
    report = coverage_report(ctx)
    assert not report.passes()
    assert report.unsupported_enabled == [offender.split(".")[0].lower()]
    ctx.set_options(tolerance=1e-10)
    ctx.solve()
    with pytest.raises(UnsupportedElementError) as info:
        extract_electrical(ctx, "t", "t/base")
    assert info.value.names == [offender]
    assert category(offender.split(".")[0].lower()) == "unsupported"
