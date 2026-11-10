"""Tests for the setpoint check of solved operating points."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.normalize.loads import normalize_loads
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from tests.circuits import ALL_CLASSES, HEAVY, LATERAL, write_circuit


def solved(tmp_path: Path, files: dict[str, str], *, normalise: float | None = None) -> DssContext:
    """Compile and solve at Tolerance 1e-10, after normalising loads to vminpu=``normalise``."""
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files))
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    if normalise is not None:
        normalize_loads(ctx, vminpu=normalise)
    assert ctx.solve().converged
    return ctx


def test_constant_power_loads_inside_their_window_pass(tmp_path: Path) -> None:
    report = setpoint_report(solved(tmp_path, LATERAL))  # model 1, source at 1.02 pu
    assert report.passes()
    assert report.n_checked == 2
    assert report.max_rel < 1e-9


def test_a_nan_setpoint_is_off(tmp_path: Path) -> None:
    ctx = solved(tmp_path, LATERAL)
    ctx.api.Loads.Name("l3")
    ctx.api.Loads.kW(float("nan"))  # NaN compares False with any tolerance
    report = setpoint_report(ctx)
    assert "Load.l3" in report.off
    assert not report.passes()


def test_normalised_window_is_enforced(tmp_path: Path) -> None:
    # vminpu 0.9: the wye loads at bus b see 0.875 pu on phase 1; the delta load (phase to phase)
    # and the load at bus a (0.93 pu) stay inside.
    report = setpoint_report(solved(tmp_path, HEAVY, normalise=0.9))
    assert report.off == ("Load.b1", "Load.bz", "Load.bi")
    assert report.worst == "Load.b1"


def test_generator_outside_its_window_is_off(tmp_path: Path) -> None:
    ctx = solved(tmp_path, HEAVY, normalise=0.6)
    # Default generator window [0.9, 1.1]; phase 1 of bus b is below 0.9 pu.
    ctx.command("New Generator.g bus1=b phases=3 kV=0.4 kW=10 kvar=0 model=1")
    assert ctx.solve().converged
    report = setpoint_report(ctx)
    assert report.off == ("Generator.g",)
    assert report.n_checked == 6


def test_generator_setpoint_sign(tmp_path: Path) -> None:
    report = setpoint_report(solved(tmp_path, ALL_CLASSES, normalise=0.6))
    assert report.passes()
    assert report.n_checked == 5  # 4 loads and the generator; PV and storage are not checked


def test_preconditions(tmp_path: Path) -> None:
    ctx = solved(tmp_path, LATERAL)
    ctx.set_options(tolerance=1e-6, max_iterations=200)
    with pytest.raises(ValueError, match="too loose"):
        setpoint_report(ctx)
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    ctx.command("Set LoadMult=0.5")
    with pytest.raises(ValueError, match="LoadMult"):
        setpoint_report(ctx)
    ctx.command("Set LoadMult=1")
    ctx.command("Set Mode=Daily")
    with pytest.raises(ValueError, match="Snapshot"):
        setpoint_report(ctx)
