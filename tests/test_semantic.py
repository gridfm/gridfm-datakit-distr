"""Tests for the semantic layer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.extract.extract_semantic import extract_semantic
from gridfm_datakit_distr.model.semantic import Bus, Line, SemanticModel, Settings
from gridfm_datakit_distr.normalize.loads import normalize_loads
from tests.circuits import (
    ALL_CLASSES,
    CONTROLLED,
    FOUR_WIRE,
    FOUR_WIRE_GROUND_OHM,
    MV_LV_RNEUT,
    RNEUT_OHM,
    SPLIT_PHASE,
    TWO_BUS,
    write_circuit,
)

UNITS = {
    "master.dss": """\
Clear
New Circuit.units basekv=0.4 pu=1.0 phases=3 bus1=src
New Line.cm bus1=src bus2=a phases=3 r1=0.001 x1=0.001 r0=0.001 x0=0.001 c1=0 c0=0
~ length=500 units=cm
New Line.in bus1=a bus2=b phases=3 r1=0.001 x1=0.001 r0=0.001 x0=0.001 c1=0 c0=0
~ length=100 units=in
New Line.none bus1=b bus2=c phases=3 r1=0.001 x1=0.001 r0=0.001 x0=0.001 c1=0 c0=0 length=2
New Load.l bus1=c phases=3 kV=0.4 kW=1 kvar=0 model=1
Set VoltageBases=[0.4]
CalcVoltageBases
""",
}


def semantic(tmp_path: Path, files: dict[str, str]) -> SemanticModel:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files))
    normalize_loads(ctx)
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    assert ctx.solve().converged
    return extract_semantic(ctx, extract_electrical(ctx, "t", "t/base"), s_base_va=1e6)


def z_matrix(line: Line) -> np.ndarray:
    return np.array(line.rs) + 1j * np.array(line.xs)


def test_line_from_sequence_impedances(tmp_path: Path) -> None:
    line = semantic(tmp_path, TWO_BUS).line["l1"]  # r1 0.1, x1 0.08, r0 0.3, x0 0.2 ohm/km, 1 km
    assert (line.length, line.length_unit) == (1000.0, "m")
    z1, z0 = 0.1 + 0.08j, 0.3 + 0.2j
    zs, zm = (z0 + 2 * z1) / 3, (z0 - z1) / 3
    expected = (np.full((3, 3), zm) + np.eye(3) * (zs - zm)) / 1000  # ohm per metre
    np.testing.assert_allclose(z_matrix(line), expected, rtol=1e-12)
    assert not np.any(line.b_fr)
    assert not np.any(line.g_to)


def test_lengths_in_metres_when_the_unit_is_known(tmp_path: Path) -> None:
    lines = semantic(tmp_path, UNITS).line
    for name, (length, unit, z_total) in {
        "cm": (5.0, "m", 0.5), "in": (2.54, "m", 0.1), "none": (2.0, "none", 0.002),
    }.items():  # fmt: skip
        assert lines[name].length == pytest.approx(length, rel=1e-12), name
        assert lines[name].length_unit == unit, name
        assert lines[name].rs[0][0] * lines[name].length == pytest.approx(z_total, rel=1e-12)


def test_grounding_through_transformer_neutral(tmp_path: Path) -> None:
    sem = semantic(tmp_path, MV_LV_RNEUT)
    lv = sem.bus["lv"]
    assert lv.grounded == [4]
    # Rneut + jXneut themselves, not the Yprim row sum (5 + j0.01006).
    assert complex(lv.rg[0], lv.xg[0]) == pytest.approx(RNEUT_OHM, rel=1e-12)
    t = sem.transformer["t"]
    assert t.connections == [[1, 2, 3], [1, 2, 3, 4]]  # no dummy neutral on the delta winding
    assert t.configuration == ["DELTA", "WYE"]
    assert sem.load["house"].connections == [1, 4]


def test_grounding_reactor_and_source_neutral(tmp_path: Path) -> None:
    sem = semantic(tmp_path, FOUR_WIRE)
    src = sem.bus["src"]
    assert (src.terminals, src.grounded) == ([1, 2, 3, 4, 5], [4, 5])
    assert src.rg == pytest.approx([FOUR_WIRE_GROUND_OHM.real, 0.0], rel=1e-9)
    assert src.xg == pytest.approx([FOUR_WIRE_GROUND_OHM.imag, 0.0], rel=1e-9)
    assert sem.shunt == {}  # the grounding reactor became bus grounding
    source = sem.voltage_source["source"]
    assert source.connections == [1, 2, 3, 5]
    assert source.vm == pytest.approx([0.4 / math.sqrt(3)] * 3 + [0.0])
    assert source.va == pytest.approx([0.0, -120.0, 120.0, 0.0])
    assert sem.load["delta"].connections == [1, 2, 3]


def test_regulator_capacitor_and_switch(tmp_path: Path) -> None:
    sem = semantic(tmp_path, CONTROLLED)
    reg = sem.transformer["reg"]
    assert reg.controls is not None
    assert (reg.controls.vreg, reg.controls.band, reg.controls.ptratio) == (
        [[0.0], [122.0]], [[0.0], [2.0]], [[0.0], [60.0]],
    )  # fmt: skip
    assert reg.tm_fix == [[True], [False]]
    assert sem.switch["sw"].state == "CLOSED"
    assert "off" not in sem.line  # disabled
    cap = sem.shunt["cap"]
    assert cap.model == "CAPACITOR"
    assert cap.bs == [[pytest.approx(100e3 / 7200.0**2, rel=1e-9)]]


PARALLEL_SWITCH = """\
Clear
New Circuit.t basekv=0.4 pu=1.0 phases=3 bus1=src
New Line.l1 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 phases=4 r1=0.1 x1=0.05 r0=0.3 x0=0.15 c1=0 c0=0
~ length=1 units=km
New Line.sw bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 phases=4 switch=yes
New Line.l2 bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 phases=4 r1=0.1 x1=0.05 r0=0.3 x0=0.15 c1=0 c0=0
~ length=1 units=km
New Reactor.g bus1=src.4 phases=1 R=0.01 X=0
New Load.ld bus1=b2.1.4 phases=1 kV=0.23 kW=5 kvar=1 model=1
{open}
Set VoltageBases=[0.4]
CalcVoltageBases
"""


@pytest.mark.parametrize("command", ["Open Line.sw 2", "Open Line.sw 2 1"])
def test_an_open_switch_keeps_its_impedance(tmp_path: Path, command: str) -> None:
    # An open conductor's Yprim rows are zero: the impedance cannot come from the Yprim.
    closed = semantic(tmp_path / "c", {"master.dss": PARALLEL_SWITCH.format(open="")})
    opened = semantic(tmp_path / "o", {"master.dss": PARALLEL_SWITCH.format(open=command)})
    a, b = closed.switch["sw"], opened.switch["sw"]
    assert (a.state, b.state) == ("CLOSED", "OPEN")
    za, zb = (np.array(s.rs) + 1j * np.array(s.xs) for s in (a, b))
    np.testing.assert_allclose(zb, za, rtol=1e-12)


def test_split_phase_service(tmp_path: Path) -> None:
    sem = semantic(tmp_path, SPLIT_PHASE)
    assert sem.transformer["ct"].connections == [[1, 4], [1, 3], [3, 2]]
    x = sem.bus["x"]
    assert (x.terminals, x.grounded) == ([1, 2, 3], [3])
    assert x.rg == pytest.approx([0.01], rel=1e-9)
    assert sem.load["l120"].connections == [1, 3]
    assert sem.load["l240"].connections == [1, 2]


def test_every_supported_class(tmp_path: Path) -> None:
    sem = semantic(tmp_path, ALL_CLASSES)
    assert set(sem.line) == {"geo", "mat", "series"}
    assert sem.line["series"].length_unit == "none"  # series reactor: lumped
    t3w = sem.transformer["t3w"]
    assert t3w.connections == [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3]]
    assert t3w.xsc == pytest.approx([0.07, 0.08, 0.03])
    assert t3w.vm_nom == [12.47, 4.16, 0.48]
    assert sem.shunt["shunt"].model == "REACTOR"
    delta = sem.shunt["cdelta"]
    assert (delta.configuration, delta.connections) == ("DELTA", [1, 2, 3])
    assert np.abs(np.sum(delta.bs, axis=1)).max() < 1e-12  # a delta bank has no path to ground
    assert sum(sem.generator["g"].pg) == pytest.approx(100.0)
    assert sum(sem.solar["pv"].pg_ub) == pytest.approx(80.0)  # Pmpp 100 kW x irradiance 0.8
    assert (sem.storage["st"].energy, sem.storage["st"].energy_ub) == (100.0, 200.0)
    assert sem.settings.vbases_default == {"src": pytest.approx(12.47 / math.sqrt(3))}
    assert SemanticModel.model_validate_json(sem.model_dump_json()) == sem


def test_validation_rejects_inconsistent_data() -> None:
    with pytest.raises(ValidationError, match="grounded terminals"):
        Bus(terminals=[1, 2, 3], grounded=[4], rg=[0.0], xg=[0.0])
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValidationError, match="rs must be 2x2"):
        Line(source_id="line.l", f_bus="a", t_bus="b", f_connections=[1, 2], t_connections=[1, 2],
             rs=[[0.0]], xs=zeros, g_fr=zeros, b_fr=zeros, g_to=zeros, b_to=zeros, length=1.0,
             length_unit="m")  # fmt: skip
    settings = Settings(sbase_default=1.0, vbases_default={}, base_frequency=50.0)
    line = Line(source_id="line.l", f_bus="a", t_bus="b", f_connections=[1], t_connections=[1],
                rs=[[1.0]], xs=[[1.0]], g_fr=[[0.0]], b_fr=[[0.0]], g_to=[[0.0]], b_to=[[0.0]],
                length=1.0, length_unit="m")  # fmt: skip
    with pytest.raises(ValidationError, match="unknown bus 'b'"):
        SemanticModel(name="v", settings=settings, bus={"a": Bus(terminals=[1])}, line={"l": line})
    with pytest.raises(ValidationError, match="no terminals"):
        SemanticModel(name="v", settings=settings, line={"l": line},
                      bus={"a": Bus(terminals=[1]), "b": Bus(terminals=[2])})  # fmt: skip
