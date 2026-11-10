"""Tests for conductor labels (split phase) and phase consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.extract.injections import injections
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.normalize.neutral import classify_neutral
from gridfm_datakit_distr.normalize.phases import PhaseError, check_phases, phase_consistency
from gridfm_datakit_distr.validate.residual import residual
from tests.circuits import ALL_CLASSES, SPLIT_PHASE, TWO_BUS, write_circuit


def extracted(
    tmp_path: Path, files: dict[str, str], extra: tuple[str, ...] = ()
) -> ElectricalModel:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files), extra_commands=extra)
    ctx.set_options(tolerance=1e-10, max_iterations=100)
    assert ctx.solve().converged
    return extract_electrical(ctx, "t", "t/base")


def labels(model: ElectricalModel) -> dict[str, str]:
    return {n.opendss_node: n.conductor for n in model.nodes}


def test_split_phase_with_explicit_neutral(tmp_path: Path) -> None:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, SPLIT_PHASE))
    ctx.set_options(tolerance=1e-10)
    assert ctx.solve().converged
    model = extract_electrical(ctx, "t", "t/base")
    got = labels(model)
    assert [got[f"{bus}.{k}"] for bus in ("x", "h") for k in (1, 2, 3)] == ["s1", "s2", "n"] * 2
    assert got["p.1"] == "a"
    assert {got[f"src.{k}"] for k in (1, 2, 3)} == {"a", "b", "c"}
    neutral = classify_neutral(model)
    assert neutral.representation == "explicit"
    assert set(neutral.grounded_neutrals) == {"x.3"}
    assert phase_consistency(model).passes()
    assert residual(model, ctx.ynode_voltages(), injections(ctx, model), tolerance=1e-10).passes()


def test_split_phase_with_grounded_centre(tmp_path: Path) -> None:
    got = labels(extracted(tmp_path, ALL_CLASSES))
    assert (got["ct.1"], got["ct.2"]) == ("s1", "s2")
    assert got["m.1"] == "a"


def test_inconsistent_lateral_is_reported(tmp_path: Path) -> None:
    lateral = (
        "New Line.bad bus1=load.2 bus2=x.1 phases=1 r1=0.1 x1=0.1 r0=0.1 x0=0.1 c1=0 c0=0"
        " length=0.1 units=km",
        "New Load.xl bus1=x.1 phases=1 kV=0.23 kW=1 kvar=0 model=1",
    )
    report = phase_consistency(extracted(tmp_path, TWO_BUS, lateral))
    assert report.mismatches == ("Line.bad: load.2 (b) -> x.1 (a)",)
    assert check_phases(report) == list(report.mismatches)  # a warning by default
    with pytest.raises(PhaseError, match="1 phase problems"):
        check_phases(report, strict=True)


def test_islanded_node_is_reported(tmp_path: Path) -> None:
    report = phase_consistency(
        extracted(
            tmp_path, TWO_BUS, ("New Load.iso bus1=iso.1 phases=1 kV=0.23 kW=1 kvar=0 model=2",)
        )
    )
    assert report.islanded == ("iso.1",)
    assert not report.passes()


def test_consistent_circuits_pass(tmp_path: Path) -> None:
    assert phase_consistency(extracted(tmp_path, TWO_BUS)).passes()
