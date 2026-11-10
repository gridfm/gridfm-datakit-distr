"""Tests for the neutral classification and grounding detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import extract_electrical
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.normalize.neutral import NeutralError, check_neutral, classify_neutral
from tests.circuits import (
    ADVERSARIAL,
    ALL_CLASSES,
    CONTROLLED,
    FOUR_WIRE,
    FOUR_WIRE_GROUND_OHM,
    LATERAL,
    MV_LV_FLOATING,
    MV_LV_MIXED,
    MV_LV_RNEUT,
    RNEUT_OHM,
    TWO_BUS,
    write_circuit,
)


def extracted(tmp_path: Path, files: dict[str, str]) -> ElectricalModel:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files))
    ctx.set_options(tolerance=1e-10, max_iterations=500)
    assert ctx.solve().converged
    return extract_electrical(ctx, "t", "t/base")


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (TWO_BUS, "kron"),
        (LATERAL, "kron"),
        (CONTROLLED, "kron"),
        (ADVERSARIAL, "kron"),
        (ALL_CLASSES, "kron"),
        (FOUR_WIRE, "explicit"),
        (MV_LV_RNEUT, "explicit"),
    ],
    ids=[
        "two_bus",
        "lateral",
        "controlled",
        "adversarial",
        "all_classes",
        "four_wire",
        "mv_lv_rneut",
    ],
)
def test_representation(tmp_path: Path, files: dict[str, str], expected: str) -> None:
    report = classify_neutral(extracted(tmp_path, files))
    assert report.representation == expected
    check_neutral(report, expected)  # type: ignore[arg-type]


def test_reactor_grounding_of_the_source_neutral(tmp_path: Path) -> None:
    report = classify_neutral(extracted(tmp_path, FOUR_WIRE))
    assert set(report.grounded_neutrals) == {"src.4"}
    assert report.grounded_neutrals["src.4"] == pytest.approx(FOUR_WIRE_GROUND_OHM, rel=1e-9)
    assert report.floating_islands == ()


def test_transformer_rneut_grounding(tmp_path: Path) -> None:
    model = extracted(tmp_path, MV_LV_RNEUT)
    report = classify_neutral(model)
    assert set(report.grounded_neutrals) == {"lv.4"}
    # Rneut sits inside the transformer Yprim; the anti-float shunt adds a few ppm.
    assert report.grounded_neutrals["lv.4"] == pytest.approx(RNEUT_OHM, rel=1e-3)
    zones = {z.buses: z.kind for z in report.zones}
    assert zones[("mv", "mv2")] == "three_wire"
    assert zones[("c", "lv")] == "explicit"


def test_floating_neutral_is_an_error(tmp_path: Path) -> None:
    report = classify_neutral(extracted(tmp_path, MV_LV_FLOATING))
    assert report.representation == "explicit"
    assert report.floating_islands == (("c.4", "lv.4"),)
    with pytest.raises(NeutralError, match="floating neutral"):
        check_neutral(report)
    check_neutral(report, allow_floating=True)


def test_mixed_is_an_error_by_default(tmp_path: Path) -> None:
    report = classify_neutral(extracted(tmp_path, MV_LV_MIXED))
    assert report.representation == "mixed"
    with pytest.raises(NeutralError, match="mixed"):
        check_neutral(report)
    check_neutral(report, allow_mixed=True)


def test_unexpected_representation_fails_c6(tmp_path: Path) -> None:
    report = classify_neutral(extracted(tmp_path, TWO_BUS))
    with pytest.raises(NeutralError, match="expected explicit"):
        check_neutral(report, "explicit")
