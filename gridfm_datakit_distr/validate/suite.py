"""Conformance suite: checks C1-C7 per network, plus voltage bases
and base-case setpoints. C8 (cross-engine) comes with the PMD exporter in the next version.

Every network gets a result: a failure inside preparation is recorded as an error with its type
and message, never dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gridfm_datakit_distr.extract.injections import injections, pc_source_admittance
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, PrepareOptions, prepare_network
from gridfm_datakit_distr.normalize.neutral import NeutralError, check_neutral
from gridfm_datakit_distr.validate.consistency import compare_admittance
from gridfm_datakit_distr.validate.golden import compare_golden, golden_path, load_golden
from gridfm_datakit_distr.validate.residual import residual

Status = Literal["pass", "fail", "skip"]
C3_REL_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class Check:
    id: str
    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class NetworkValidation:
    network_id: str
    checks: tuple[Check, ...]
    error: str | None = None
    """Exception type and message if the network could not be prepared."""
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.status != "fail" for c in self.checks)


def _status(ok: bool) -> Status:
    return "pass" if ok else "fail"


def run_checks(prepared: PreparedNetwork, golden_dir: Path | None = None) -> tuple[Check, ...]:
    """Checks C1-C7, voltage bases and setpoints of a prepared network."""
    ctx, model, network = prepared.ctx, prepared.model, prepared.network
    v = ctx.ynode_voltages()
    solve = prepared.solve
    checks = [
        Check(
            "C1",
            "base case solves",
            _status(solve.converged),
            f"{solve.iterations} iterations, Tolerance {prepared.options.tolerance:g}",
        )
    ]

    r = residual(model, v, injections(ctx, model), tolerance=prepared.options.tolerance)
    checks.append(
        Check(
            "C2",
            "KCL residual",
            _status(r.passes()),
            f"ratio {r.ratio:.1e} at {r.worst_node}, rel {r.rel:.1e}, backward {r.backward:.1e}",
        )
    )

    y = compare_admittance(model, model.ybus(), ctx.system_y(), pc_source_admittance(ctx, model))
    checks.append(
        Check(
            "C3",
            "Y_state vs solver Y",
            _status(y.passes(C3_REL_TOL)),
            f"max rel {y.max_rel:.1e} at {y.worst[0]}/{y.worst[1]} ({y.n_entries} entries)",
        )
    )

    unsupported = prepared.coverage.unsupported_enabled
    checks.append(
        Check(
            "C4",
            "element coverage",
            _status(not unsupported),
            f"unsupported enabled: {unsupported}" if unsupported else "all supported",
        )
    )

    ph = prepared.phases
    checks.append(
        Check(
            "C5",
            "phase consistency",
            _status(ph.passes()),
            f"{len(ph.mismatches)} label mismatches, {len(ph.islanded)} islanded nodes",
        )
    )

    try:
        check_neutral(prepared.neutral, network.neutral)
        checks.append(
            Check(
                "C6",
                "neutral representation",
                "pass",
                f"{prepared.neutral.representation} (expected {network.neutral})",
            )
        )
    except NeutralError as exc:
        checks.append(Check("C6", "neutral representation", "fail", str(exc)))

    golden = load_golden(golden_dir, network.id) if golden_dir is not None else None
    if golden_dir is None:
        checks.append(Check("C7", "golden file", "skip", "no golden directory given"))
    elif golden is None:  # a wrong path or a schema bump must not pass unnoticed
        checks.append(
            Check("C7", "golden file", "fail", f"missing {golden_path(golden_dir, network.id)}")
        )
    else:
        g = compare_golden(model, v, golden)
        checks.append(Check("C7", "golden file", _status(g.passed), g.detail))

    b = prepared.bases
    checks.append(
        Check(
            "bases",
            "voltage bases",
            _status(b.passes()),
            f"{len(prepared.base_fix)} buses re-based; max deviation "
            f"{b.max_deviation:.1e}; ambiguous {len(b.ambiguous)}, "
            f"conflicts {len(b.conflicts)}",
        )
    )
    s = prepared.setpoints
    checks.append(
        Check(
            "setpoints",
            "base-case setpoints",
            _status(s.passes()),
            f"{len(s.off)} of {s.n_checked} off; max deviation {s.max_rel:.1e}",
        )
    )
    return tuple(checks)


def validate_network(
    network: Network,
    *,
    root: Path | None = None,
    golden_dir: Path | None = None,
    options: PrepareOptions | None = None,
) -> NetworkValidation:
    """Prepare ``network`` and run the suite; any exception becomes a recorded error."""
    start = time.perf_counter()
    try:
        prepared = prepare_network(network, options, root)
        checks = run_checks(prepared, golden_dir)
    except Exception as exc:  # recorded, never dropped
        message = str(exc).splitlines()[0] if str(exc) else ""
        return NetworkValidation(
            network.id, (), f"{type(exc).__name__}: {message}", time.perf_counter() - start
        )
    return NetworkValidation(network.id, checks, None, time.perf_counter() - start)
