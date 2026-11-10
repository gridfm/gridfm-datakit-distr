"""Tests for the dry-run estimates and the dataset summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.config.value_spec import parse_spec
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.solve.estimate import BYTES, SECONDS, estimate_network, mean_of
from gridfm_datakit_distr.solve.executor import generate
from gridfm_datakit_distr.stats.compute import summary, summary_text
from tests.circuits import VARIANT_FEEDER, write_circuit
from tests.test_executor import catalog, config


def test_mean_of_a_spec() -> None:
    assert mean_of(parse_spec(0.3)) == pytest.approx(0.3, rel=1e-12)
    uniform = parse_spec({"distribution": "uniform", "low": 0.0, "high": 0.4})
    assert mean_of(uniform) == pytest.approx(0.2, abs=0.01)


def test_estimate_counts_the_rows(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", VARIANT_FEEDER)
    prepared = prepare_network(
        Network(id="t/net", source="src", master="master.dss"), root=tmp_path
    )
    cfg = GenerationConfig.model_validate(
        {"dataset_name": "t", "networks": {"include": ["x"]}, "der": {"pv": {"penetration": 0.5}},
         "scenarios": {"n_scenarios_per_variant": 1}, "settings": {"seed": 1}}
    )  # fmt: skip
    e = estimate_network(prepared, cfg)
    assert e.nodes == len(prepared.model.nodes)
    assert e.customers == 6
    assert e.kron_nodes == sum(n.conductor != "n" for n in prepared.model.nodes)
    assert not e.controls
    assert e.seconds_per_scenario == pytest.approx(
        SECONDS[0] + SECONDS[1] * e.nodes + SECONDS[2] * 6
    )
    b = BYTES["mixed"]
    floor = b["meta"] + b["node"] * e.nodes + b["load"] * 6
    assert e.bytes_per_scenario > floor  # plus PV and the positive-sequence view (default on)
    assert e.bytes_per_variant > 0


def test_summary_of_a_dataset(tmp_path: Path) -> None:
    out = tmp_path / "out"
    generate(config(), out, catalog=catalog(tmp_path), root=tmp_path)
    s = summary(out)
    assert s.units == {"done": 12}
    assert sum(sum(c.values()) for c in s.scenarios.values()) == 20
    assert s.views == {}  # no view by default
    text = summary_text(s)
    assert text.startswith("dataset d (schema ")
    assert "t/reg:" in text
