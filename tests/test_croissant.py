"""Tests for the Croissant metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.croissant import (
    CONFORMS_TO,
    PLACEHOLDER,
    RAI_PLACEHOLDERS,
    croissant,
    validate,
    write_croissant,
)
from gridfm_datakit_distr.solve.executor import generate
from tests.test_executor import catalog, config


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("croissant")
    out = root / "out"
    generate(config(), out, catalog=catalog(root), root=root)
    return out


def test_the_executor_writes_it(dataset: Path) -> None:
    doc = json.loads((dataset / "croissant.json").read_text(encoding="utf-8"))
    assert doc == croissant(dataset)  # deterministic: no date, no machine
    assert doc["conformsTo"] == CONFORMS_TO
    assert doc["name"] == "d"
    assert doc["license"] == ["https://spdx.org/licenses/CC0-1.0.html"]  # the test source's


def test_record_sets_follow_the_schema(dataset: Path) -> None:
    doc = croissant(dataset)
    sets = {r["name"]: r for r in doc["recordSet"]}
    assert {"scenarios/node_results", "states/states"} <= set(sets)
    assert not any(name.startswith("views/") for name in sets)  # no view by default
    for name, record_set in sets.items():
        expected = (
            schema.SCENARIO_TABLES[name]("mixed") if name in schema.SCENARIO_TABLES
            else schema.TABLES[name]
        )  # fmt: skip
        assert [f["name"] for f in record_set["field"]] == expected.names, name
    files = {d["@id"]: d for d in doc["distribution"]}
    states = files["states/states-files"]
    digest = hashlib.sha256((dataset / "states" / "states.parquet").read_bytes()).hexdigest()
    assert (states["@type"], states["sha256"]) == ("cr:FileObject", digest)
    assert files["scenarios/node_results-files"]["includes"] == "scenarios/node_results/*.parquet"
    assert "passports-files" in files
    assert "semantic-files" in files
    assert all(doc[k] == PLACEHOLDER for k in RAI_PLACEHOLDERS)
    assert doc["rai:dataCollectionType"] == "Simulation"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # rdflib, inside mlcroissant
def test_it_passes_the_official_validator(dataset: Path) -> None:
    """Validation with mlcroissant (needs the 'croissant' extra, outside the dev environment)."""
    mlc = pytest.importorskip("mlcroissant")
    path = write_croissant(dataset)
    warnings = validate(path)
    assert all("datePublished" in w for w in warnings)  # recommended; a date breaks determinism
    loaded = mlc.Dataset(jsonld=str(path))
    meta = list(loaded.records(record_set="scenarios/scenario_meta"))
    assert len(meta) == 20
