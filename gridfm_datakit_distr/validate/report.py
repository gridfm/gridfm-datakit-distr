"""Validation reports: ``validation/<network_id>.json`` and a Markdown summary."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gridfm_datakit_distr.io.schema import SCHEMA_VERSION
from gridfm_datakit_distr.validate.suite import NetworkValidation

COLUMNS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "bases", "setpoints")
MARK = {"pass": "ok", "fail": "FAIL", "skip": "-"}


def result_json(result: NetworkValidation) -> dict[str, Any]:
    return {
        "network_id": result.network_id,
        "schema_version": SCHEMA_VERSION,
        "passed": result.passed,
        "error": result.error,
        "seconds": round(result.seconds, 3),
        "checks": [asdict(c) for c in result.checks],
    }


def markdown_summary(results: Sequence[NetworkValidation]) -> str:
    """One row per network; reasons for failures and errors listed under the table."""
    failed = [r for r in results if not r.passed]
    lines = [
        "# Conformance summary",
        "",
        f"{len(results)} networks: {len(results) - len(failed)} pass, {len(failed)} fail "
        f"(schema {SCHEMA_VERSION}). Checks: docs/manual/networks.md; `-` = skipped.",
        "",
        "| network | " + " | ".join(COLUMNS) + " | result |",
        "|---|" + "---|" * len(COLUMNS) + "---|",
    ]
    for r in results:
        by_id = {c.id: MARK[c.status] for c in r.checks}
        cells = [by_id.get(column, "error" if r.error else "-") for column in COLUMNS]
        verdict = "pass" if r.passed else "**FAIL**"
        lines.append(f"| {r.network_id} | " + " | ".join(cells) + f" | {verdict} |")
    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            if r.error:
                lines.append(f"- {r.network_id}: error: {r.error}")
            for c in r.checks:
                if c.status == "fail":
                    lines.append(f"- {r.network_id}: {c.id} ({c.name}): {c.detail}")
    return "\n".join(lines) + "\n"


def write_reports(results: Sequence[NetworkValidation], out_dir: Path) -> Path:
    """Write one JSON per network and ``summary.md``; return the summary path."""
    for r in results:
        path = out_dir / f"{r.network_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result_json(r), indent=2) + "\n", encoding="utf-8")
    summary = out_dir / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(markdown_summary(results), encoding="utf-8")
    return summary
