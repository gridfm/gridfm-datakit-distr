"""Julia environment for the OPF cross-checks (juliacall + juliapkg).

The Julia side is declared in ``gridfm_datakit_distr/juliapkg.json`` and resolved by juliapkg on
the first ``import juliacall`` (Julia itself is downloaded if no compatible one is found). Only
``opf/`` and the Julia tests import this module: the power-flow path never does.
Requires the ``julia`` extra.
"""

from __future__ import annotations

import time
from typing import Any

JULIA_PACKAGES = ("PowerModelsDistribution", "Ipopt", "JSON")
REPORTED_INDIRECT = ("JuMP", "InfrastructureModels", "MathOptInterface", "Ipopt_jll")


def julia_main() -> Any:
    """``juliacall.Main``, resolving and installing the Julia environment on first use."""
    try:
        from juliacall import Main
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "Julia support needs the 'julia' extra: uv sync --extra julia "
            "(or pip install 'gridfm-datakit-distr[julia]')"
        ) from exc
    return Main


def load_packages() -> Any:
    """Load the Julia packages of :data:`JULIA_PACKAGES` (precompiling them if needed)."""
    jl = julia_main()
    jl.seval(f"using {', '.join(JULIA_PACKAGES)}")
    return jl


def versions() -> dict[str, str]:
    """Versions of Julia and of the main packages of the resolved Julia environment."""
    jl = load_packages()
    jl.seval("import Pkg")
    deps = jl.seval("Dict(string(i.name) => string(i.version) for i in values(Pkg.dependencies()))")
    out = {"julia": str(jl.seval("string(VERSION)"))}
    for name in (*JULIA_PACKAGES, *REPORTED_INDIRECT):
        out[name] = str(deps[name]) if name in deps else "absent"
    return out


def setup() -> dict[str, float | str]:
    """Resolve, install and precompile the Julia environment; return versions and timings (s)."""
    t0 = time.perf_counter()
    julia_main()
    t1 = time.perf_counter()
    load_packages()
    t2 = time.perf_counter()
    report: dict[str, float | str] = dict(versions())
    report["seconds_resolve_and_start"] = t1 - t0
    report["seconds_load_packages"] = t2 - t1
    return report
