"""Smoke tests for the package skeleton and the CLI entry point."""

import re

from typer.testing import CliRunner

import gridfm_datakit_distr
from gridfm_datakit_distr.cli import app


def test_version_is_pep440_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]?(a|b|rc|dev)\d+)?", gridfm_datakit_distr.__version__)


def test_cli_version_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == gridfm_datakit_distr.__version__
