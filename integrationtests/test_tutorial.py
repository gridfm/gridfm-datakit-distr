"""The getting started guide runs as written: its CLI commands on IEEE 13, its Python snippets."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gridfm_datakit_distr.cli import app

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]
REPO = Path(__file__).parents[1]
TUTORIAL = REPO / "docs" / "manual" / "getting_started.md"


def blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", TUTORIAL.read_text(encoding="utf-8"), re.S)


def test_the_tutorial_runs(
    network_master: Callable[[str], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    network_master("ieee13")
    runner = CliRunner()
    config = str(REPO / "scripts" / "config" / "ieee13_small.yaml")
    monkeypatch.chdir(tmp_path)  # the snippets read data/ieee13_small, as in the tutorial
    for args in (
        ["generate", config, "--dry-run"],
        ["generate", config, "--data-dir", "data"],
        ["stats", "data/ieee13_small"],
        ["export-posseq", "data/ieee13_small", "--out", "data/ieee13_small_datakit"],
    ):
        assert any(" ".join(args[:1]) in b for b in blocks("bash"))
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    namespace: dict[str, Any] = {}
    snippets = blocks("python")
    assert len(snippets) >= 5
    for i, code in enumerate(snippets):
        exec(compile(code, f"tutorial-snippet-{i}", "exec"), namespace)
    assert namespace["kcl"] < 1e-6


def test_the_example_configs_validate() -> None:
    from gridfm_datakit_distr.config.load import load_config

    for path in sorted((REPO / "scripts" / "config").glob("*.yaml")):
        load_config(path)  # raises ConfigError with every issue
