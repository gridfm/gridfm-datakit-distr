"""Statistics of a CSIRO dataset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridfm_datakit_distr.cli import app
from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.stats.physics import physics

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]


def test_a_csiro_dataset(network_master: Callable[[str], Path], tmp_path: Path) -> None:
    network_master("csiro_enwl/n01_f1")
    cfg = GenerationConfig.model_validate(
        {
            "dataset_name": "csiro",
            "networks": {"include": ["csiro_enwl/n01_f1"]},
            "variants": {
                "n_variants": 1,
                "admittance": {"line_length_scale": 1.0},
                "phase_reallocation": {"enabled": False},
            },  # unperturbed, as before the defaults
            "der": {"pv": {"penetration": 0.3}},
            "views": {"kron": False},
            "scenarios": {"n_scenarios_per_variant": 12},
            "settings": {"seed": 4, "chunk_size": 6, "num_processes": 1},
        }
    )
    out = tmp_path / "csiro"
    generate(cfg, out)
    stats = physics(out, 0.94, 1.10)["csiro_enwl/n01_f1"]
    assert stats.scenarios == 12
    vuf, neutral = stats.histograms["vuf"], stats.histograms["neutral_v"]
    assert vuf.n == 12 * 907  # every bus of the feeder is three-phase
    assert 0 < vuf.quantile(0.5) < vuf.maximum < 0.05
    assert neutral.n > 0
    assert 0 < neutral.maximum < 0.1  # neutral rise on a four-wire LV feeder (pu of 230 V)
    result = CliRunner().invoke(app, ["stats", str(out), "--plots"])
    assert result.exit_code == 0, result.output
    assert (out / "stats" / "stats.json").exists()
    assert (out / "stats" / "vuf.png").exists()
    assert (out / "stats" / "neutral_voltage.png").exists()
