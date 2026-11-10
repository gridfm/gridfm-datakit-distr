"""The CLI on the example config."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridfm_datakit_distr.cli import app
from gridfm_datakit_distr.io.manifest import Manifest

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]
CONFIG = Path(__file__).parents[1] / "scripts" / "config" / "ieee13_small.yaml"


def test_generate_dry_run_resume_and_stats(
    network_master: Callable[[str], Path], tmp_path: Path
) -> None:
    network_master("ieee13")
    runner = CliRunner()
    args = ["generate", str(CONFIG), "--data-dir", str(tmp_path), "--processes", "1"]
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "1 network(s) x 2 variant(s) x 20 scenarios = 40 scenarios in 4 units" in dry.output
    assert "ieee13: 41 nodes, 15 customers" in dry.output
    assert "note: no explicit neutral in ieee13" in dry.output  # neutral_representation: explicit
    assert not (tmp_path / "ieee13_small").exists()  # nothing written
    run = runner.invoke(app, args)
    assert run.exit_code == 0, run.output
    assert "scenarios: 40" in run.output
    dataset = tmp_path / "ieee13_small"
    assert {u.status for u in Manifest.load(dataset).units.values()} == {"done"}
    out = str(tmp_path / "dk")
    export = runner.invoke(app, ["export-posseq", str(dataset), "--out", out, "--processes", "2"])
    assert export.exit_code == 0, export.output
    assert "ieee13: 40 scenarios in" in export.output
    assert "(views: converged 40)" in export.output
    raw = tmp_path / "dk" / "ieee13" / "raw"
    assert (raw / "n_scenarios.txt").read_text(encoding="utf-8") == "40"
    again = runner.invoke(app, args)  # resume: nothing left to do, same dataset
    assert again.exit_code == 0, again.output
    stats = runner.invoke(app, ["stats", str(dataset)])
    assert stats.exit_code == 0, stats.output
    assert "units: done 4" in stats.output
    other = runner.invoke(app, [*args[:-2], "--processes", "2"])  # execution settings only
    assert other.exit_code == 0, other.output
