"""Tests for the OpenDSS engine context wrapper."""

from __future__ import annotations

import gc
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.engine.context import (
    DssContext,
    SolveStatus,
    complex_from_interleaved,
    matrix_from_yprim,
)
from gridfm_datakit_distr.engine.errors import DssError
from tests.circuits import LATERAL, TWO_BUS, TWO_BUS_Z_LINE_OHM, TWO_BUS_Z_LOAD_OHM, write_circuit


@pytest.fixture
def two_bus(tmp_path: Path) -> Path:
    return write_circuit(tmp_path / "two_bus", TWO_BUS)


@pytest.fixture
def lateral(tmp_path: Path) -> Path:
    return write_circuit(tmp_path / "lateral", LATERAL)


def _loaded(master: Path) -> DssContext:
    ctx = DssContext.new()
    ctx.load(master)
    ctx.set_options(tolerance=1e-10)
    return ctx


# ---------------------------------------------------------------- layout helpers


def test_complex_from_interleaved() -> None:
    out = complex_from_interleaved(np.array([1.0, 2.0, 3.0, -4.0]))
    np.testing.assert_array_equal(out, np.array([1 + 2j, 3 - 4j]))


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 2)), np.zeros(4, dtype=np.complex128)])
def test_complex_from_interleaved_rejects_other_layouts(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="interleaved"):
        complex_from_interleaved(bad)


def test_matrix_from_yprim_is_column_major() -> None:
    # entries of a non-symmetric 2x2 matrix, column by column: (0,0), (1,0), (0,1), (1,1)
    flat = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype=np.float64)
    np.testing.assert_array_equal(matrix_from_yprim(flat), np.array([[1, 3], [2, 4]]))


def test_matrix_from_yprim_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        matrix_from_yprim(np.zeros(6))


# ---------------------------------------------------------------- context


def test_two_bus_solution_matches_analytic_ratio(two_bus: Path) -> None:
    ctx = _loaded(two_bus)
    status = ctx.solve()
    assert status.converged
    voltages = dict(zip(ctx.ynode_order(), ctx.ynode_voltages(), strict=True))
    expected = TWO_BUS_Z_LOAD_OHM / (TWO_BUS_Z_LINE_OHM + TWO_BUS_Z_LOAD_OHM)
    for k in (1, 2, 3):
        ratio = voltages[f"load.{k}"] / voltages[f"src.{k}"]
        assert abs(ratio - expected) < 1e-9 * abs(expected)


def test_ynode_order_is_lower_case_bus_dot_conductor(two_bus: Path) -> None:
    ctx = _loaded(two_bus)
    assert ctx.ynode_order() == ["src.1", "src.2", "src.3", "load.1", "load.2", "load.3"]


def test_ynode_positions_are_looked_up_by_name_in_any_case(two_bus: Path) -> None:
    ctx = _loaded(two_bus)
    assert ctx.ynode_positions(["LOAD.2", "src.1"]).tolist() == [4, 0]
    with pytest.raises(KeyError):
        ctx.ynode_positions(["nowhere.1"])


def test_line_yprim_matches_sequence_impedances(two_bus: Path) -> None:
    z1, z0 = 0.1 + 0.08j, 0.3 + 0.2j  # ohm (1 km)
    zs, zm = (2 * z1 + z0) / 3, (z0 - z1) / 3
    y = np.linalg.inv(np.full((3, 3), zm) + np.eye(3) * (zs - zm))
    expected = np.block([[y, -y], [-y, y]])
    yprim = _loaded(two_bus).element_yprim("Line.l1")
    assert yprim.shape == (6, 6)
    np.testing.assert_allclose(yprim, expected, rtol=1e-12, atol=1e-12)


def test_missing_element_raises(two_bus: Path) -> None:
    ctx = _loaded(two_bus)
    with pytest.raises(DssError, match="element not found"):
        ctx.element_yprim("Line.does_not_exist")


@pytest.mark.parametrize(
    ("cmd", "code"),
    [("bogus_command", 302), ("New Line.x bus1=src bus2=y bogusprop=3", 110)],
)
def test_invalid_command_raises_dss_error(two_bus: Path, cmd: str, code: int) -> None:
    ctx = _loaded(two_bus)
    with pytest.raises(DssError) as info:
        ctx.command(cmd)
    assert info.value.code == code
    assert info.value.command == cmd


def test_command_returns_result_text(two_bus: Path) -> None:
    ctx = _loaded(two_bus)
    assert float(ctx.command("? Line.l1.length")) == 1.0


def test_load_keeps_working_directory_and_resolves_relative_includes(
    two_bus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    ctx = DssContext.new()
    ctx.load(two_bus)
    assert Path(os.getcwd()) == elsewhere
    ctx.activate_element("Line.l1")  # defined in lines/line.dss, included relatively


def test_load_missing_master_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DssContext.new().load(tmp_path / "missing.dss")


def test_load_runs_calcvoltagebases_only_when_needed(tmp_path: Path) -> None:
    files = dict(TWO_BUS)
    master = write_circuit(tmp_path / "with", files)
    ctx = DssContext.new()
    ctx.load(master)
    assert ctx.load_actions == []

    files["master.dss"] = files["master.dss"].replace("CalcVoltageBases\n", "")
    master = write_circuit(tmp_path / "without", files)
    ctx.load(master)
    assert ctx.load_actions == ["CalcVoltageBases"]
    assert ctx.buses_without_voltage_base() == []


def test_set_options_round_trip(two_bus: Path) -> None:
    ctx = DssContext.new()
    ctx.load(two_bus)
    ctx.set_options(tolerance=1e-10, max_iterations=50, control_mode="OFF")
    assert ctx.api.Solution.Convergence() == pytest.approx(1e-10)
    assert ctx.api.Solution.MaxIterations() == 50
    assert ctx.api.Solution.ControlMode() == -1
    ctx.set_options(control_mode="STATIC")
    assert ctx.api.Solution.ControlMode() == 0


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"tolerance": 0.0}, "tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"control_mode": "TIME"}, "control_mode"),
    ],
)
def test_set_options_rejects_invalid_values(
    two_bus: Path, kwargs: dict[str, object], field: str
) -> None:
    ctx = DssContext.new()
    ctx.load(two_bus)
    with pytest.raises(ValueError, match=field):
        ctx.set_options(**kwargs)  # type: ignore[arg-type]


def test_non_convergence_is_reported_not_raised(lateral: Path) -> None:
    ctx = DssContext.new()
    ctx.load(lateral)
    ctx.set_options(tolerance=1e-12, max_iterations=1)
    assert ctx.solve() == SolveStatus(converged=False, iterations=1)


def test_engine_version_mentions_dss_c_api(two_bus: Path) -> None:
    version = DssContext.new().engine_version()
    assert "DSS C-API" in version
    assert "\n" not in version


def test_a_dropped_context_clears_its_circuit(two_bus: Path) -> None:
    # DSS-Python never disposes a context, so a dropped one must free its circuit itself.
    ctx = _loaded(two_bus)
    api = ctx.api
    assert api.Circuit.NumBuses() > 0
    del ctx
    gc.collect()
    assert api.Basic.NumCircuits() == 0


# ---------------------------------------------------------------- concurrency


def _solve_sequence(
    master: Path, load: str, kws: list[float], barrier: threading.Barrier | None = None
) -> list[np.ndarray]:
    ctx = DssContext.new()
    ctx.load(master)
    ctx.set_options(tolerance=1e-10)
    if barrier is not None:
        barrier.wait(timeout=30)
    results = []
    for kw in kws:
        ctx.command(f"Edit {load} kW={kw}")
        assert ctx.solve().converged
        results.append(ctx.ynode_voltages().copy())
    return results


def test_two_contexts_in_two_threads_solve_concurrently(two_bus: Path, lateral: Path) -> None:
    kws_a = [5.0 + 0.5 * k for k in range(200)]
    kws_b = [200.0 + 5.0 * k for k in range(200)]
    reference_a = _solve_sequence(two_bus, "Load.ld", kws_a)
    reference_b = _solve_sequence(lateral, "Load.l3", kws_b)

    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_solve_sequence, two_bus, "Load.ld", kws_a, barrier)
        future_b = pool.submit(_solve_sequence, lateral, "Load.l3", kws_b, barrier)
        concurrent_a, concurrent_b = future_a.result(), future_b.result()

    # Same inputs, same engine, same operation order: results must be bit-identical.
    for got, ref in zip(concurrent_a, reference_a, strict=True):
        np.testing.assert_array_equal(got, ref)
    for got, ref in zip(concurrent_b, reference_b, strict=True):
        np.testing.assert_array_equal(got, ref)
