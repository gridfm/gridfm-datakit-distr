"""Croissant metadata of a dataset.

``croissant.json`` describes the dataset in MLCommons Croissant 1.0 (JSON-LD): one file set per
partitioned table (``<table>/*.parquet``), one file object with its SHA-256 per single-file table,
one record set per table with the fields of its schema (``io/schema.py``), the passports and
semantic layers as JSON file sets, licences and citations from the passports, and the
Responsible-AI fields: those this generator knows (how the data were made), the others as
placeholders for the publisher. Nothing in it depends on the date or the machine, so it is the
same for the same dataset. Validation (``validate``) needs the optional ``mlcroissant``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.manifest import Manifest, atomic_write_text

CONFORMS_TO = "http://mlcommons.org/croissant/1.0"
CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}
"""The Croissant 1.0 context (as in the specification and mlcroissant's ``make_context``)."""
PARQUET = "application/x-parquet"
PLACEHOLDER = "To be completed by the publisher before release."
RAI_KNOWN = {
    "rai:dataCollection": (
        "Simulated. Published distribution network models (see each network's passport: source, "
        "licence, recipe) are perturbed into variants and solved in OpenDSS for sampled loads, "
        "rooftop PV and source voltages (gridfm-datakit-distr)."
    ),
    "rai:dataCollectionType": "Simulation",
}
RAI_PLACEHOLDERS = (
    "rai:dataCollectionMissingData",
    "rai:dataCollectionRawData",
    "rai:dataPreprocessingProtocol",
    "rai:dataBiases",
    "rai:dataUseCases",
    "rai:dataLimitations",
    "rai:dataSocialImpact",
    "rai:personalSensitiveInformation",
    "rai:dataReleaseMaintenancePlan",
)


def _data_type(t: pa.DataType) -> str:
    if pa.types.is_integer(t):
        return "sc:Integer"
    if pa.types.is_floating(t):
        return "sc:Float"
    if pa.types.is_boolean(t):
        return "sc:Boolean"
    return "sc:Text"


def _tables(manifest: Manifest) -> dict[str, pa.Schema]:
    """Every table of the contract and its schema in this dataset's precision."""
    precision = manifest.config["settings"]["float_precision"]
    tables = {name: build(precision) for name, build in schema.SCENARIO_TABLES.items()}
    tables.update(schema.TABLES)
    return tables


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _license(text: str) -> str:
    """An SPDX identifier as its URL; anything else as given."""
    if text and " " not in text:
        return f"https://spdx.org/licenses/{text}.html"
    return text


def croissant(root: Path) -> dict[str, Any]:
    """The Croissant metadata of the dataset in ``root``."""
    manifest = Manifest.load(root)
    config = manifest.config
    passports = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((root / "catalog" / "passports").rglob("*.json"))
    ]
    distribution: list[dict[str, Any]] = []
    record_sets: list[dict[str, Any]] = []
    for name, table_schema in _tables(manifest).items():
        single, parts = root / f"{name}.parquet", root / name
        file_id = f"{name}-files"
        if single.is_file():
            distribution.append({
                "@type": "cr:FileObject", "@id": file_id, "name": file_id,
                "contentUrl": f"{name}.parquet", "encodingFormat": PARQUET,
                "sha256": _sha256(single),
            })  # fmt: skip
            reference = {"fileObject": {"@id": file_id}}
        elif parts.is_dir() and any(parts.glob("*.parquet")):
            distribution.append({
                "@type": "cr:FileSet", "@id": file_id, "name": file_id,
                "encodingFormat": PARQUET, "includes": f"{name}/*.parquet",
            })  # fmt: skip
            reference = {"fileSet": {"@id": file_id}}
        else:
            continue
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": name,
                "name": name,
                "description": f"Table {name} (docs/manual/outputs.md of gridfm-datakit-distr).",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": f"{name}/{f.name}",
                        "name": f.name,
                        "dataType": _data_type(f.type),
                        "source": {**reference, "extract": {"column": f.name}},
                    }
                    for f in table_schema
                ],
            }
        )
    for name, pattern, what in (
        ("passports", "catalog/passports/**/*.json", "network passports"),
        ("semantic", "variants/semantic/**/*.json", "semantic layers of the variants"),
    ):
        if any(root.glob(pattern)):
            distribution.append({
                "@type": "cr:FileSet", "@id": f"{name}-files", "name": f"{name}-files",
                "description": what, "encodingFormat": "application/json", "includes": pattern,
            })  # fmt: skip
    networks = [n["network_id"] for n in manifest.networks]
    description = (
        f"Power-flow dataset of {len(networks)} three-phase unbalanced distribution network(s) "
        f"({', '.join(networks)}), {config['variants']['n_variants']} variant(s) each, "
        f"{config['scenarios']['n_scenarios_per_variant']} scenarios per variant, generated by "
        f"gridfm-datakit-distr {manifest.versions.get('generator', '?')} with OpenDSS "
        f"(schema {manifest.schema_version}; tables in the generator's "
        "docs/manual/outputs.md). The "
        "generator's command export-posseq writes its positive-sequence equivalent in "
        "gridfm-datakit's layout."
    )
    doc: dict[str, Any] = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CONFORMS_TO,
        "name": config["dataset_name"],
        "description": description,
        "version": manifest.schema_version,
        "license": sorted({_license(str(p["license"])) for p in passports}),
        "citeAs": "\n\n".join(sorted({str(p["citation"]) for p in passports if p.get("citation")})),
        "keywords": ["power flow", "distribution network", "three-phase", "unbalanced", "OpenDSS"],
        "distribution": distribution,
        "recordSet": record_sets,
        **RAI_KNOWN,
        **dict.fromkeys(RAI_PLACEHOLDERS, PLACEHOLDER),
    }
    return doc


def write_croissant(root: Path) -> Path:
    """Write ``croissant.json`` in the dataset; returns its path."""
    path = root / "croissant.json"
    atomic_write_text(path, json.dumps(croissant(root), indent=1, ensure_ascii=False) + "\n")
    return path


def validate(path: Path) -> list[str]:
    """Validate with mlcroissant (raises ``mlcroissant.ValidationError`` on errors); returns
    its warnings. Needs the optional ``mlcroissant``."""
    import mlcroissant as mlc

    dataset = mlc.Dataset(jsonld=str(path))
    return [str(w) for w in dataset.metadata.ctx.issues.warnings]
