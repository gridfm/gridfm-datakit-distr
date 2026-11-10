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
    assert element(model, "Line.l1").rating_a == pytest.approx(400.0)  # OpenDSS default NormAmps


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
