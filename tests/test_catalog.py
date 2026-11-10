"""Tests for the network catalog and manifests."""

from __future__ import annotations

import re
from typing import Any

import pytest

from gridfm_datakit_distr.networks.catalog import (
    CatalogError,
    build_catalog,
    format_manifest,
    load_catalog,
    parse_manifest,
)

H1, H2, H3 = "1" * 64, "2" * 64, "3" * 64
MATCH = (
    r"^(?P<representation>[^/]+)/[Nn]etwork_(?P<network>\d+)"
    r"/Feeder_(?P<feeder>\d+)/Master\.dss$"
)


def raw_catalog(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "sources": {
            "src": {
                "kind": "csiro_dap", "title": "t", "homepage": "h", "license": "l",
                "citation": "c", "collection": 1, "version": 1,
            }
        },
        "networks": [{"id": "net", "source": "src", "master": "single/Master.dss"}],
        "collections": [
            {
                "source": "src", "id": "src/n{network:02d}_f{feeder}",
                "match": MATCH,
                "representations": {"Four": "four_wire", "Kron": "kron"},
                "default": "four_wire", "neutral": "explicit",
            }
        ],
    }  # fmt: skip
    raw.update(overrides)
    return raw


MANIFEST = {
    "single/Master.dss": H1,
    "Four/network_3/Feeder_1/Master.dss": H1,
    "Kron/Network_3/Feeder_1/Master.dss": H2,
    "Four/network_3/Feeder_1/Lines.txt": H3,
}


# ---------------------------------------------------------------- manifests


def test_manifest_round_trip_is_canonical() -> None:
    entries = {"b/x.dss": H1, "a.dss": H2}
    text = format_manifest(entries)
    assert text == f"{H2}  a.dss\n{H1}  b/x.dss\n"
    assert parse_manifest("# comment\n\n" + text) == entries


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (f"{H1[:-1]}  a.dss\n", "expected"),
        (f"{H1} a.dss\n", "expected"),
        (f"{H1}  ../a.dss\n", "unsafe"),
        (f"{H1}  /abs.dss\n", "unsafe"),
        (f"{H1}  a.dss\n{H2}  a.dss\n", "duplicate"),
    ],
)
def test_bad_manifests_are_rejected(text: str, message: str) -> None:
    with pytest.raises(CatalogError, match=message):
        parse_manifest(text)


# ---------------------------------------------------------------- catalog building


def test_collection_expands_one_network_per_feeder() -> None:
    catalog = build_catalog(raw_catalog(), {"src": MANIFEST})
    feeder = catalog.network("src/n03_f1")
    assert feeder.master == "Four/network_3/Feeder_1/Master.dss"
    assert feeder.neutral == "explicit"
    assert feeder.representations == {
        "four_wire": "Four/network_3/Feeder_1/Master.dss",
        "kron": "Kron/Network_3/Feeder_1/Master.dss",
    }
    assert sorted(catalog.networks) == ["net", "src/n03_f1"]


def test_source_for_resolves_sources_and_networks() -> None:
    catalog = build_catalog(raw_catalog(), {"src": MANIFEST})
    assert catalog.source_for("src").id == "src"
    assert catalog.source_for("src/n03_f1").id == "src"
    with pytest.raises(KeyError, match="neither a source nor a network"):
        catalog.source_for("nope")


def test_master_must_be_in_the_manifest() -> None:
    manifest = {k: v for k, v in MANIFEST.items() if k != "single/Master.dss"}
    with pytest.raises(CatalogError, match="not in the manifest"):
        build_catalog(raw_catalog(), {"src": manifest})


def test_duplicate_ids_unknown_sources_and_folders_are_rejected() -> None:
    dup = raw_catalog(
        networks=[{"id": "src/n03_f1", "source": "src", "master": "single/Master.dss"}]
    )
    with pytest.raises(CatalogError, match="duplicate network id"):
        build_catalog(dup, {"src": MANIFEST})
    unknown = raw_catalog(networks=[{"id": "x", "source": "nope", "master": "single/Master.dss"}])
    with pytest.raises(CatalogError, match="unknown source"):
        build_catalog(unknown, {"src": MANIFEST})
    with pytest.raises(CatalogError, match="unknown representation folder"):
        build_catalog(raw_catalog(), {"src": {**MANIFEST, "Odd/network_1/Feeder_1/Master.dss": H1}})


def test_unknown_fields_are_rejected() -> None:
    raw = raw_catalog()
    raw["sources"]["src"]["colection"] = 2
    with pytest.raises(CatalogError, match="colection"):
        build_catalog(raw, {"src": MANIFEST})


def test_manifest_fingerprint_ignores_order() -> None:
    a = build_catalog(raw_catalog(), {"src": MANIFEST}).source("src")
    b = build_catalog(raw_catalog(), {"src": dict(reversed(list(MANIFEST.items())))}).source("src")
    assert a.manifest_sha256 == b.manifest_sha256


# ---------------------------------------------------------------- packaged catalog


def test_packaged_catalog_has_the_planned_networks() -> None:
    catalog = load_catalog()
    for network_id in ("ieee13", "ieee34", "ieee37", "ieee123", "ieee_eulv", "ieee8500"):
        assert catalog.source_for(network_id).id == "epri_ieee"
        assert catalog.network(network_id).neutral == "kron"
    for member in ("industrial", "rural", "urban"):
        assert catalog.source_for(f"edp/{member}").id == "edp"
    assert set(catalog.source("edp").manifest) == {"Industrial.rar", "Rural.rar", "Urban.rar"}
    fixes = catalog.network("ieee34").extra_commands  # the delta-load fix
    assert len(fixes) == 6
    pattern = r"Edit Load\.D832_858[sr][abc] bus1=8(32|58)\.([123])\.(?!\2)[123]"
    assert all(re.fullmatch(pattern, c) for c in fixes)  # two distinct phases each
    assert catalog.network("ieee13").extra_commands == []


def test_packaged_csiro_collection_has_128_four_wire_feeders() -> None:
    catalog = load_catalog()
    feeders = [n for n in catalog.networks.values() if n.source == "csiro_enwl"]
    assert len(feeders) == 128
    for feeder in feeders:
        assert feeder.neutral == "explicit"
        assert feeder.master.startswith("Four-wire/")
        assert set(feeder.representations) == {
            "four_wire", "kron", "phase_to_neutral", "modified_phase_to_neutral"
        }  # fmt: skip
