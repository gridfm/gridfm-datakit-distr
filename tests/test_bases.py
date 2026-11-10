"""Tests for per-unit bases and the voltage-base check."""

from __future__ import annotations

import cmath
import math
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.units import SQRT3, Bases, Quantity
from gridfm_datakit_distr.normalize.bases import (
    BaseError,
    BaseReport,
    check_bases,
    fix_bases,
    s_base_va,
)
from tests.circuits import ALL_CLASSES, CONTROLLED, SPLIT_PHASE, TWO_BUS, mv_lv, write_circuit

base_value = st.floats(1e-3, 1e9)
part = st.one_of(st.just(0.0), st.floats(1e-6, 1e6), st.floats(-1e6, -1e-6))
phasor = st.builds(complex, part, part)
QUANTITIES: tuple[Quantity, ...] = get_args(Quantity)


@given(s=base_value, v=base_value)
def test_bases_are_consistent(s: float, v: float) -> None:
    b = Bases(s, v)
    assert math.isclose(b.z_base_ohm * b.i_base_a, b.v_ln_v, rel_tol=1e-12)
    assert math.isclose(3 * b.v_ln_v * b.i_base_a, s, rel_tol=1e-12)
    assert math.isclose(b.y_base_s * b.z_base_ohm, 1.0, rel_tol=1e-12)
    assert math.isclose(Bases.from_ln(s, b.v_ln_v).v_ll_v, v, rel_tol=1e-12)


@given(s=base_value, v=base_value, x=phasor)
def test_per_unit_round_trip(s: float, v: float, x: complex) -> None:
    b = Bases(s, v)
    for q in QUANTITIES:
        assert cmath.isclose(b.from_pu(b.to_pu(x, q), q), x, rel_tol=1e-12)
        assert cmath.isclose(b.to_pu(b.from_pu(x, q), q), x, rel_tol=1e-12)


@given(s=base_value, v=base_value, z=phasor.filter(lambda c: c != 0), i=phasor)
def test_circuit_laws_hold_in_per_unit(s: float, v: float, z: complex, i: complex) -> None:
    b = Bases(s, v)
    u = z * i  # phase-to-neutral voltage across z
    z_pu, i_pu, u_pu = b.to_pu(z, "impedance"), b.to_pu(i, "current"), b.to_pu(u, "voltage")
    assert cmath.isclose(u_pu, z_pu * i_pu, rel_tol=1e-9)
    assert cmath.isclose(
        b.to_pu(3 * u * i.conjugate(), "power"), u_pu * i_pu.conjugate(), rel_tol=1e-9
    )
    assert cmath.isclose(b.to_pu(1 / z, "admittance"), 1 / z_pu, rel_tol=1e-9)


def test_per_unit_arrays() -> None:
    b = Bases(1e6, 400.0)
    v = np.array([230.0 + 1j, 0.0, -115.0 - 199.2j])
    assert b.from_pu(b.to_pu(v, "voltage"), "voltage") == pytest.approx(v, rel=1e-15)
    assert b.to_pu(np.array([400.0 / SQRT3]), "voltage") == pytest.approx([1.0], rel=1e-15)


def test_bases_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Bases(0.0, 400.0)


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
