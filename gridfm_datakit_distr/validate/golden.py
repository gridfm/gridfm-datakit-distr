"""Golden files for check C7.

A golden file holds derived numbers only: the SHA-256 of the extracted model's
structure (node names, conductor labels, voltage bases, grounding and source flags; element names,
classes and terminal maps) and the base-case node voltages in pu of their bus bases, in node
order (the hash fixes that order; no source identifier is stored). Files are gzip-compressed JSON
with a zero timestamp, so the same content gives the same bytes. C7 passes
when the structure hash is equal and every voltage is within ``GOLDEN_TOL_PU`` of the golden one.
The numbers of the Yprims enter through the voltages, so the hash does not depend on round-off.
Files live under ``<golden_dir>/<schema_version>/<network_id>.json.gz``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.io.schema import SCHEMA_VERSION
from gridfm_datakit_distr.model.electrical import ElectricalModel

GOLDEN_TOL_PU = 1e-9
GOLDEN_DECIMALS = 12
"""Stored voltages are rounded to 1e-12 pu, three orders below the C7 tolerance."""


def structure_sha256(model: ElectricalModel) -> str:
    """SHA-256 of the model's structure (no Yprim numbers)."""
    payload = {
        "nodes": [
            [n.opendss_node, n.conductor, f"{n.kv_base_ln:.9g}", n.is_grounded, n.is_source]
            for n in model.nodes
        ],
        "elements": [[e.name.lower(), e.element_class, e.terminals] for e in model.elements],
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def voltages_pu(model: ElectricalModel, voltages: NDArray[np.complex128]) -> NDArray[np.complex128]:
    bases = np.array([n.kv_base_ln * 1e3 for n in model.nodes])
    return np.asarray(voltages / bases, dtype=np.complex128)


def golden_record(model: ElectricalModel, voltages: NDArray[np.complex128]) -> dict[str, Any]:
    """The golden file content of a solved base case."""
    v = voltages_pu(model, voltages)
    return {
        "network_id": model.network_id,
        "schema_version": SCHEMA_VERSION,
        "structure_sha256": structure_sha256(model),
        "v_re": [round(float(x), GOLDEN_DECIMALS) + 0.0 for x in v.real],
        "v_im": [round(float(x), GOLDEN_DECIMALS) + 0.0 for x in v.imag],
    }


def golden_path(golden_dir: Path, network_id: str) -> Path:
    return golden_dir / SCHEMA_VERSION / f"{network_id}.json.gz"


def load_golden(golden_dir: Path, network_id: str) -> dict[str, Any] | None:
    path = golden_path(golden_dir, network_id)
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return data


def write_golden(golden_dir: Path, record: dict[str, Any]) -> Path:
    path = golden_path(golden_dir, record["network_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as f:
        f.write((json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8"))
    path.write_bytes(buffer.getvalue())
    return path


@dataclass(frozen=True, slots=True)
class GoldenComparison:
    passed: bool
    detail: str
    max_dev_pu: float | None = None


def compare_golden(
    model: ElectricalModel, voltages: NDArray[np.complex128], golden: dict[str, Any]
) -> GoldenComparison:
    """Check C7 against a golden record."""
    if golden["structure_sha256"] != structure_sha256(model):
        return GoldenComparison(False, "model structure differs from the golden file")
    stored = np.array(golden["v_re"]) + 1j * np.array(golden["v_im"])
    dev = np.abs(voltages_pu(model, voltages) - stored)
    worst = int(np.argmax(dev))
    max_dev = float(dev[worst])
    detail = f"max |dV| {max_dev:.1e} pu at {model.nodes[worst].opendss_node}"
    return GoldenComparison(max_dev <= GOLDEN_TOL_PU, detail, max_dev)
