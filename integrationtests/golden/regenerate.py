"""Regenerate the C7 golden files.

Run after an intended change, review the diff, commit:

    uv run python integrationtests/golden/regenerate.py         # the reference set (networks.txt)
    uv run python integrationtests/golden/regenerate.py ieee13     # selected networks

A golden file is written only if the network passes every other check of the suite: a golden
reference must not record a wrong state.
Files: ``integrationtests/golden/<schema_version>/<id>.json.gz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.validate.golden import golden_record, write_golden
from gridfm_datakit_distr.validate.suite import run_checks

GOLDEN_DIR = Path(__file__).parent
NETWORKS = Path(__file__).with_name("networks.txt").read_text().split()
"""The reference set: IEEE, every sixth CSIRO feeder by id plus the four used in the conformance
tests (n13_f4 is excluded), the three EDP networks."""


def main(ids: list[str]) -> int:
    catalog = load_catalog()
    status = 0
    for network_id in ids:
        prepared = prepare_network(catalog.network(network_id))
        failed = [c for c in run_checks(prepared) if c.status == "fail"]
        if failed:
            print(f"{network_id}: NOT written, failing {[f'{c.id}: {c.detail}' for c in failed]}")
            status = 1
            continue
        path = write_golden(
            GOLDEN_DIR, golden_record(prepared.model, prepared.ctx.ynode_voltages())
        )
        print(f"{network_id}: {path.relative_to(GOLDEN_DIR)}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or NETWORKS))
