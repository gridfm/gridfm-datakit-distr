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
OPF_SECTION = "## 5. OPF labels"


def blocks(language: str, *, opf: bool = False) -> list[str]:
    """The code blocks of the guide outside its OPF section (``opf``: of the OPF section only)."""
    text = TUTORIAL.read_text(encoding="utf-8")
    head, _, tail = text.partition(OPF_SECTION)
    opf_part, _, rest = tail.partition("\n## ")
    return re.findall(rf"```{language}\n(.*?)```", opf_part if opf else head + rest, re.S)


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


def test_the_opf_example_runs(
    network_master: Callable[[str], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("casadi")
    network_master("csiro_enwl/n02_f5")
    config = str(REPO / "scripts" / "config" / "opf_small.yaml")
    monkeypatch.chdir(tmp_path)
    args = ["generate", config, "--data-dir", "data"]
    assert any("generate scripts/config/opf_small.yaml" in b for b in blocks("bash", opf=True))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    namespace: dict[str, Any] = {}
    (snippet,) = blocks("python", opf=True)
    exec(compile(snippet, "tutorial-opf", "exec"), namespace)
    assert namespace["solved"]
    assert namespace["curtailed"]


def test_the_example_configs_validate() -> None:
    from gridfm_datakit_distr.config.load import load_config

    for path in sorted((REPO / "scripts" / "config").glob("*.yaml")):
        load_config(path)  # raises ConfigError with every issue
