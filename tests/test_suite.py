"""Tests for the conformance suite, golden files and reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gridfm_datakit_distr.cli import app
from gridfm_datakit_distr.io.schema import SCHEMA_VERSION
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.validate.golden import (
    compare_golden,
    golden_path,
    golden_record,
    load_golden,
    write_golden,
)
from gridfm_datakit_distr.validate.report import markdown_summary, write_reports
from gridfm_datakit_distr.validate.suite import run_checks, validate_network
from tests.circuits import FOUR_WIRE, LATERAL, write_circuit


def network(tmp_path: Path, files: dict[str, str], **fields: object) -> Network:
    write_circuit(tmp_path / "src", files)
    return Network(id="t/net", source="src", master="master.dss", **fields)  # type: ignore[arg-type]


def test_a_sound_network_passes(tmp_path: Path) -> None:
    result = validate_network(network(tmp_path, LATERAL, neutral="kron"), root=tmp_path)
    assert result.passed
    assert result.error is None
    assert [c.id for c in result.checks] == [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "bases", "setpoints",
    ]  # fmt: skip
    assert {c.id: c.status for c in result.checks}["C7"] == "skip"  # no golden directory


def test_a_wrong_expectation_fails_with_its_reason(tmp_path: Path) -> None:
    result = validate_network(network(tmp_path, FOUR_WIRE, neutral="kron"), root=tmp_path)
    assert not result.passed
    failed = [c for c in result.checks if c.status == "fail"]
    assert [c.id for c in failed] == ["C6"]
    assert "explicit, expected kron" in failed[0].detail


def test_errors_are_recorded_not_raised(tmp_path: Path) -> None:
    bad = network(tmp_path, LATERAL, master_stop_before="redirect missing.dss")
    result = validate_network(bad, root=tmp_path)
    assert not result.passed
    assert result.checks == ()
    assert result.error is not None
    assert result.error.startswith("CatalogError: ")


def test_golden_file_round_trip_and_detection(tmp_path: Path) -> None:
    prepared = prepare_network(network(tmp_path, LATERAL, neutral="kron"), root=tmp_path)
    v = prepared.ctx.ynode_voltages()
    golden_dir = tmp_path / "golden"
    path = write_golden(golden_dir, golden_record(prepared.model, v))
    assert path == golden_path(golden_dir, "t/net")
    golden = load_golden(golden_dir, "t/net")
    assert golden is not None
    assert compare_golden(prepared.model, v, golden).passed
    assert {c.id: c.status for c in run_checks(prepared, golden_dir)}["C7"] == "pass"
    # a golden directory without this network's file (or the versioned folder by mistake) fails
    for wrong in (tmp_path / "none", golden_dir / SCHEMA_VERSION):
        [c7] = [c for c in run_checks(prepared, wrong) if c.id == "C7"]
        assert (c7.status, c7.detail) == ("fail", f"missing {golden_path(wrong, 't/net')}")
    base = prepared.model.nodes[-1].kv_base_ln * 1e3
    moved = v.copy()
    moved[-1] += 2e-9 * base  # 2e-9 pu
    result = compare_golden(prepared.model, moved, golden)
    assert not result.passed
    assert result.max_dev_pu == pytest.approx(2e-9, rel=1e-6)
    golden["structure_sha256"] = "0" * 64
    assert "structure" in compare_golden(prepared.model, v, golden).detail


def test_reports(tmp_path: Path) -> None:
    good = validate_network(network(tmp_path / "a", LATERAL, neutral="kron"), root=tmp_path / "a")
    bad = validate_network(network(tmp_path / "b", FOUR_WIRE, neutral="kron"), root=tmp_path / "b")
    summary = write_reports([good, bad], tmp_path / "out")
    text = summary.read_text(encoding="utf-8")
    assert text == markdown_summary([good, bad])
    assert "2 networks: 1 pass, 1 fail" in text
    assert "- t/net: C6 (neutral representation): " in text
    data = json.loads((tmp_path / "out" / "t" / "net.json").read_text(encoding="utf-8"))
    assert data["passed"] is False  # the second result overwrote the first file: same id
    assert np.isfinite(data["seconds"])


def test_validate_cli_reports_unknown_and_unfetched(tmp_path: Path) -> None:
    runner = CliRunner()
    unknown = runner.invoke(app, ["validate", "nowhere", "--cache", str(tmp_path)])
    assert unknown.exit_code == 2
    assert "unknown network or source 'nowhere'" in unknown.output
    unfetched = runner.invoke(app, ["validate", "ieee13", "--cache", str(tmp_path)])
    assert unfetched.exit_code == 1
    assert "sources not fetched: epri_ieee" in unfetched.output
    nothing = runner.invoke(app, ["validate", "all", "--cache", str(tmp_path)])
    assert nothing.exit_code == 1
    assert "no network to validate" in nothing.output
