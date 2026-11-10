"""Tests for passports and ``inspect``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridfm_datakit_distr.cli import app
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.passport import build_passport, write_passport
from gridfm_datakit_distr.networks.prepare import prepare_network

SNAPSHOT = Path(__file__).parent / "snapshots" / "inspect_ieee13.txt"
REQUIRED = {  # outputs.md §3
    "network_id", "source_url", "license", "citation", "checksums", "original_format",
    "converter", "normalisation", "neutral_representation", "slack_definition",
    "known_differences", "excluded_elements",
}  # fmt: skip


def test_inspect_rejects_unknown_networks() -> None:
    result = CliRunner().invoke(app, ["inspect", "no_such_network"])
    assert result.exit_code == 2
    assert "unknown network 'no_such_network'" in result.output


def test_inspect_needs_the_fetched_source(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", "ieee13", "--cache", str(tmp_path)])
    assert result.exit_code == 1
    assert "fetch epri_ieee" in result.output


@pytest.mark.network_data
def test_inspect_ieee13_snapshot(network_master: Callable[[str], Path], tmp_path: Path) -> None:
    network_master("ieee13")  # skips without the cache
    result = CliRunner().invoke(app, ["inspect", "ieee13", "--passport-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    summary, _, written = result.output.rpartition("passport: ")
    # To refresh after an intended change: gridfm_datakit_distr inspect ieee13 > <SNAPSHOT>
    assert summary == SNAPSHOT.read_text(encoding="utf-8")
    passport = json.loads(Path(written.strip()).read_text(encoding="utf-8"))
    assert set(passport) >= REQUIRED
    assert passport["checksums"]["master_sha256"] is not None
    assert all(passport["checks"].values())


@pytest.mark.network_data
def test_passport_records_the_recipe(network_master: Callable[[str], Path], tmp_path: Path) -> None:
    network_master("ieee34")
    catalog = load_catalog()
    passport = build_passport(prepare_network(catalog.network("ieee34")), catalog)
    assert len(passport["recipe"]["extra_commands"]) == 6
    assert passport["known_differences"][0].startswith("source model compiled with corrections")
    path = write_passport(passport, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == passport
