"""Tests for the electrical-layer extractor on inline circuits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import (
    UnsupportedElementError,
    disabled_elements,
    extract_electrical,
)
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from tests.circuits import (
    ADVERSARIAL,
    CONTROLLED,
    FOUR_WIRE,
    FOUR_WIRE_GROUND_OHM,
    TWO_BUS,
    write_circuit,
)


def extract(tmp_path: Path, files: dict[str, str], extra: tuple[str, ...] = ()) -> ElectricalModel:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files), extra_commands=extra)
    ctx.set_options(tolerance=1e-10)
    assert ctx.solve().converged
    return extract_electrical(ctx, "test", "test/base")


def by_name(model: ElectricalModel) -> dict[str, int]:
    return {n.opendss_node: n.idx for n in model.nodes}


def element(model: ElectricalModel, name: str) -> Element:
    (found,) = [e for e in model.elements if e.name.lower() == name.lower()]
    return found


def test_two_bus_nodes_terminals_and_source(tmp_path: Path) -> None:
    model = extract(tmp_path, TWO_BUS)
    idx = by_name(model)
    assert sorted(idx) == ["load.1", "load.2", "load.3", "src.1", "src.2", "src.3"]
    assert {n.conductor for n in model.nodes} == {"a", "b", "c"}
    assert all(abs(n.kv_base_ln - 0.4 / np.sqrt(3)) < 1e-12 for n in model.nodes)
    assert sorted(model.nodes[i].opendss_node for i in model.source_nodes()) == [
        "src.1",
        "src.2",
        "src.3",
    ]

    line = element(model, "Line.l1")
    assert (line.element_class, line.is_pd, line.enabled) == ("line", True, True)
    assert line.terminals == [[idx["src.1"], idx["src.2"], idx["src.3"]],
                              [idx["load.1"], idx["load.2"], idx["load.3"]]]  # fmt: skip
    assert line.yprim is not None
    assert line.yprim.shape == (6, 6)

    load = element(model, "Load.ld")
    assert (load.is_pd, load.yprim) == (False, None)
    assert load.terminals == [[idx["load.1"], idx["load.2"], idx["load.3"], GROUND]]
    source = element(model, "Vsource.source")
    assert source.terminals[1] == [GROUND, GROUND, GROUND]


def test_four_wire_neutral_grounding_and_delta(tmp_path: Path) -> None:
    model = extract(tmp_path, FOUR_WIRE)
    idx = by_name(model)
    assert {"src.4", "b1.4"} <= set(idx)
    assert model.nodes[idx["b1.4"]].conductor == "n"
    neutral = model.nodes[idx["src.4"]]
    assert neutral.is_grounded
    assert not neutral.is_source
    assert complex(neutral.r_ground_ohm, neutral.x_ground_ohm) == pytest.approx(
        FOUR_WIRE_GROUND_OHM
    )
    assert not model.nodes[idx["b1.4"]].is_grounded
    assert element(model, "Load.house").terminals == [[idx["b1.1"], idx["b1.4"]]]
    assert element(model, "Load.delta").terminals == [[idx["b1.1"], idx["b1.2"], idx["b1.3"]]]
    assert element(model, "Line.l1").yprim.shape == (8, 8)


def test_controls_switches_and_disabled_elements(tmp_path: Path) -> None:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, CONTROLLED))
    assert disabled_elements(ctx) == ["Line.off"]
    model = extract(tmp_path, CONTROLLED)
    names = {e.name.lower() for e in model.elements}
    assert names == {"vsource.source", "transformer.reg", "line.l1", "line.sw", "capacitor.cap",
                     "load.ld"}  # fmt: skip
    assert element(model, "Transformer.reg").is_tap_dependent
    assert element(model, "Capacitor.cap").is_tap_dependent
    assert not element(model, "Line.l1").is_tap_dependent
    assert element(model, "Line.sw").element_class == "switch"
    assert element(model, "Transformer.reg").rating_va == pytest.approx(1000e3)
    assert element(model, "Line.l1").rating_a is None  # no ampacity declared: not 400 A


# Lines whose ampacity is declared (on the line, its linecode or its wires) or not.
RATED = {
    "master.dss": """\
Clear
New Circuit.rated basekv=0.4 pu=1.0 phases=3 bus1=src angle=0
New Linecode.plain nphases=3 r1=0.1 x1=0.1 r0=0.3 x0=0.3 c1=0 c0=0 units=km
New Linecode.rated nphases=3 r1=0.1 x1=0.1 r0=0.3 x0=0.3 c1=0 c0=0 units=km normamps=250
New Linecode.rated400 nphases=3 r1=0.1 x1=0.1 r0=0.3 x0=0.3 c1=0 c0=0 units=km normamps=400
New Wiredata.plain GMR=0.0244 DIAM=0.721 RAC=0.306 Runits=mi radunits=in gmrunits=ft
New Wiredata.rated GMR=0.0244 DIAM=0.721 RAC=0.306 NormAmps=530 Runits=mi radunits=in gmrunits=ft
New Linegeometry.plain nconds=3 nphases=3
~ cond=1 wire=plain x=-1 h=10 units=m cond=2 wire=plain x=0 h=10 units=m
~ cond=3 wire=plain x=1 h=10 units=m
New Linegeometry.rated nconds=3 nphases=3
~ cond=1 wire=rated x=-1 h=10 units=m cond=2 wire=rated x=0 h=10 units=m
~ cond=3 wire=rated x=1 h=10 units=m
New Line.code_default bus1=src bus2=a linecode=plain length=0.1
New Line.code_rated bus1=src bus2=b linecode=rated length=0.1
New Line.code_rated400 bus1=src bus2=c linecode=rated400 length=0.1
New Line.line_400 bus1=src bus2=d linecode=plain length=0.1 normamps=400
New Line.overwritten bus1=src bus2=e normamps=300 linecode=plain length=0.1
New Line.direct bus1=src bus2=f r1=0.1 x1=0.1 r0=0.3 x0=0.3 c1=0 c0=0 length=0.1
New Line.direct_400 bus1=src bus2=g r1=0.1 x1=0.1 r0=0.3 x0=0.3 c1=0 c0=0 length=0.1 normamps=400
New Line.geo_default bus1=src bus2=h geometry=plain length=0.1 units=km
New Line.geo_rated bus1=src bus2=i geometry=rated length=0.1 units=km
New Line.sw bus1=src bus2=j switch=yes
New Load.ld bus1=a phases=3 kV=0.4 kW=10 kvar=2 model=1
Set VoltageBases=[0.4]
CalcVoltageBases
""",
}


@pytest.mark.parametrize(
    ("line", "rating"),
    [
        ("code_default", None),
        ("code_rated", 250.0),
        ("code_rated400", 400.0),
        ("line_400", 400.0),
        ("overwritten", None),  # the linecode assigned after it resets the line's value
        ("direct", None),
        ("direct_400", 400.0),
        ("geo_default", None),
        ("geo_rated", 530.0),
        ("sw", None),
    ],
)
def test_rating_a_is_null_unless_declared(tmp_path: Path, line: str, rating: float | None) -> None:
    model = extract(tmp_path, RATED)
    assert element(model, f"Line.{line}").rating_a == rating


def test_unsupported_enabled_element_raises(tmp_path: Path) -> None:
    fault = "New Fault.f1 bus1=load.1 phases=1 r=1"
    with pytest.raises(UnsupportedElementError) as info:
        extract(tmp_path / "on", TWO_BUS, (fault,))
    assert info.value.names == ["Fault.f1"]
    model = extract(tmp_path / "off", TWO_BUS, (fault, "Disable Fault.f1"))
    assert all(not e.name.lower().startswith("fault.") for e in model.elements)


def test_model_admittance_matches_line_yprim(tmp_path: Path) -> None:
    model = extract(tmp_path, TWO_BUS)
    line = element(model, "Line.l1")
    local = line.local_nodes()
    np.testing.assert_array_equal(model.ybus().toarray()[np.ix_(local, local)], line.yprim)


def test_shunt_reactor_on_a_phase_is_not_a_grounding(tmp_path: Path) -> None:
    shunt = "New Reactor.shunt phases=1 bus1=load.1 bus2=load.0 R=1 X=5"
    model = extract(tmp_path, TWO_BUS, (shunt,))
    assert not any(n.is_grounded for n in model.nodes)
    assert element(model, "Reactor.shunt").is_pd


def test_adversarial_layout_maps_every_phase(tmp_path: Path) -> None:
    model = extract(tmp_path, ADVERSARIAL)
    idx = by_name(model)
    assert {name for name in idx if name.startswith("m2.")} == {"m2.2", "m2.3"}
    assert [model.nodes[idx[n]].conductor for n in ("m2.2", "m2.3", "far.3")] == ["b", "c", "c"]
    assert element(model, "Line.far").terminals == [[idx["m2.3"]], [idx["far.3"]]]
    assert element(model, "Load.bc").terminals == [[idx["m2.2"], idx["m2.3"]]]
    assert element(model, "Load.c").terminals == [[idx["far.3"], GROUND]]
