"""Tests for per-unit bases and the voltage-base check."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.units import SQRT3
from gridfm_datakit_distr.normalize.bases import (
    BaseError,
    BaseReport,
    check_bases,
    fix_bases,
    s_base_va,
)
from tests.circuits import ALL_CLASSES, CONTROLLED, SPLIT_PHASE, TWO_BUS, mv_lv, write_circuit


def solved(tmp_path: Path, files: dict[str, str]) -> tuple[DssContext, ElectricalModel]:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files))
    ctx.set_options(tolerance=1e-10, max_iterations=100)
    assert ctx.solve().converged
    return ctx, extract_electrical(ctx, "t", "t/base")


def test_consistent_bases_pass(tmp_path: Path) -> None:
    for name, files in {"two_bus": TWO_BUS, "controlled": CONTROLLED}.items():
        ctx, model = solved(tmp_path / name, files)
        report = check_bases(ctx, model)
        assert report.passes(), name
        assert report.max_deviation < 1e-4, name  # 7.2 kV regulator vs 12.47/√3 kV: 6e-5
        assert report.fix_commands() == []


def test_wrong_split_phase_level_is_rebased(tmp_path: Path) -> None:
    ctx, model = solved(tmp_path, SPLIT_PHASE)  # VoltageBases [12.47 0.24]: 0.1386 kV LN
    report = check_bases(ctx, model)
    assert report.rebase == pytest.approx({"x": 0.12, "h": 0.12})
    assert not report.ambiguous
    assert not report.conflicts
    assert sorted(fix_bases(ctx, report)) == [
        "SetkVBase bus=h kVLN=0.12",
        "SetkVBase bus=x kVLN=0.12",
    ]
    ctx.solve()
    fixed = check_bases(ctx, extract_electrical(ctx, "t", "t/base"))
    assert fixed.passes()
    assert fixed.max_deviation < 1e-12


def test_offset_secondary_rating_is_ambiguous(tmp_path: Path) -> None:
    # 22/0.42 kV transformer, author's base 0.4 kV: -4.8 %, either may be the intended nominal.
    ctx, model = solved(tmp_path, mv_lv("Rneut=5 Xneut=0.01 kVs=[22 0.42]"))
    report = check_bases(ctx, model)
    assert not report.passes()
    assert not report.rebase
    assert not report.conflicts
    assert any(m.startswith("bus lv: ") and m.endswith("(-4.8%)") for m in report.ambiguous)
    assert fix_bases(ctx, report) == []


def test_disagreeing_ratings_in_one_zone_conflict(tmp_path: Path) -> None:
    parallel = (
        "New Transformer.t2 windings=2 Buses=[mv2.1.2.3 lv.1.2.3.4] Conns=[Delta Wye]"
        " kVs=[22 0.42] kVAs=[100 100] XHL=4"
    )
    ctx, model = solved(tmp_path, mv_lv("Rneut=5 Xneut=0.01", parallel))
    report = check_bases(ctx, model)
    assert not report.passes()
    assert len(report.conflicts) == 1
    assert "Transformer.t winding 2" in report.conflicts[0]  # the larger unit sets the nominal
    assert "Transformer.t2 winding 2" in report.conflicts[0]


def test_every_winding_kind_sets_a_zone(tmp_path: Path) -> None:
    ctx, model = solved(tmp_path, ALL_CLASSES)
    nominal = {bus: z.nominal_kv_ln for z in check_bases(ctx, model).zones for bus in z.buses}
    assert nominal["src"] == pytest.approx(12.47 / SQRT3)  # source zone (Vsource)
    assert nominal["t3a"] == pytest.approx(4.16 / SQRT3)  # 3-phase wye winding
    assert nominal["t3b"] == pytest.approx(0.48 / SQRT3)  # 3-phase delta tertiary
    assert nominal["ct"] == pytest.approx(0.12)  # split-phase halves (1-phase wye)
    assert nominal["bk"] == pytest.approx(2.4)  # bank of 1-phase wye windings


def test_fix_reads_every_base_back(tmp_path: Path) -> None:
    ctx, _ = solved(tmp_path, TWO_BUS)
    unknown = BaseReport((), {"nowhere": 0.23}, (), (), 1.0)
    with pytest.raises(BaseError, match="nowhere"):
        fix_bases(ctx, unknown)


def test_power_base(tmp_path: Path) -> None:
    ctx, model = solved(tmp_path / "ctrl", CONTROLLED)
    assert s_base_va(ctx, model) == (1000e3, "Transformer.reg rating")
    ctx, model = solved(tmp_path / "two", TWO_BUS)
    value, origin = s_base_va(ctx, model)
    assert origin == "total nominal load"
    assert value == pytest.approx(abs(10e3 + 5e3j))


def test_a_declared_nominal_voltage_reaches_its_level(tmp_path: Path) -> None:
    from gridfm_datakit_distr.networks.catalog import Network
    from gridfm_datakit_distr.networks.prepare import prepare_network

    write_circuit(tmp_path / "src", TWO_BUS)
    network = Network(id="t/two", source="src", master="master.dss", nominal_kv_ln={0.4: 0.22})
    model = prepare_network(network, root=tmp_path).model
    assert {n.kv_nominal_ln for n in model.nodes} == {0.22}
    wrong = network.model_copy(update={"nominal_kv_ln": {11.0: 6.35}})
    with pytest.raises(ValueError, match=r"no bus on the declared 11\.0 kV level"):
        prepare_network(wrong, root=tmp_path)
