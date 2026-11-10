"""Smoke test of the Julia environment: PMD parses IEEE 13 and solves its power flow.

PMD 0.16's native power flow is singular on the Kron-reduced IEEE 13; it converges in the
explicit-neutral mode used by PMD's own validation tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = [pytest.mark.julia, pytest.mark.network_data, pytest.mark.slow]


def test_pmd_parses_ieee13_and_its_power_flow_converges(
    network_master: Callable[[str], Path],
) -> None:
    from gridfm_datakit_distr.opf.julia_env import load_packages, versions

    master = network_master("ieee13")
    jl = load_packages()
    jl.PowerModelsDistribution.silence_b()  # silence!()
    run = jl.seval("""p -> begin
        PMD = PowerModelsDistribution
        eng = PMD.parse_file(p; transformations=[PMD.transform_loops!])
        math = PMD.transform_data_model(eng; kron_reduce=false)
        (eng, PMD.compute_mc_pf(math; explicit_neutral=true))
    end""")
    eng, result = run(str(master))
    assert len(eng["bus"]) == 16
    assert str(result["termination_status"]) == "PF_CONVERGED"
    assert versions()["PowerModelsDistribution"] == "0.16.0"
