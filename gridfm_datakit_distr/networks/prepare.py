"""Prepare a catalog network: compile, normalise, solve, extract and check.

Order: compile with the catalog recipe; load normalisation; element
coverage (C4); solve (C1); extraction; voltage bases (buses on a wrong level are re-based,
then the circuit is solved and extracted again, since bases enter the stopping test); neutral
(C6) and phase (C5) reports; setpoints of the base case. Hard failures (unsupported
elements, extraction errors) raise; check outcomes are reported, not raised.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from gridfm_datakit_distr.engine.context import DssContext, SolveStatus
from gridfm_datakit_distr.extract.extract_electrical import disabled_elements, extract_electrical
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.ids import base_variant_id
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.compile import CompileRecord, compile_network
from gridfm_datakit_distr.normalize.bases import BaseReport, check_bases, fix_bases, s_base_va
from gridfm_datakit_distr.normalize.coverage import CoverageReport, coverage_report
from gridfm_datakit_distr.normalize.loads import LoadModel, LoadNormalisation, normalize_loads
from gridfm_datakit_distr.normalize.neutral import NeutralReport, classify_neutral
from gridfm_datakit_distr.normalize.phases import PhaseReport, phase_consistency
from gridfm_datakit_distr.validate.setpoints import SetpointReport, setpoint_report


@dataclass(frozen=True, slots=True)
class PrepareOptions:
    """Normalisation parameters (config ``normalisation.loads``) and solver settings."""

    load_model: LoadModel = "constant_power"
    vminpu: float = 0.6
    vmaxpu: float = 1.5
    tolerance: float = 1e-10
    max_iterations: int = 200


@dataclass(frozen=True, slots=True)
class PreparedNetwork:
    """A compiled, normalised and solved network with everything its passport reports."""

    network: Network
    options: PrepareOptions
    ctx: DssContext
    model: ElectricalModel
    compiled: CompileRecord
    loads: LoadNormalisation
    coverage: CoverageReport
    solve: SolveStatus
    bases_as_loaded: BaseReport
    base_fix: tuple[str, ...]
    """``SetkVBase`` commands applied; empty if no bus was on a wrong level."""
    bases: BaseReport
    s_base: tuple[float, str]
    neutral: NeutralReport
    phases: PhaseReport
    setpoints: SetpointReport
    excluded: tuple[str, ...]
    """Disabled elements, left out of the model."""
    edits: tuple[str, ...] = ()
    """The variant's edit script; empty for the network as loaded."""


def with_nominal(model: ElectricalModel, levels: dict[float, float]) -> ElectricalModel:
    """``model`` with the nominal voltage of the levels the catalog declares (line-to-line base
    kV -> nominal phase-to-neutral kV); a declared level that matches no bus is an error."""
    if not levels:
        return model
    nodes = list(model.nodes)
    for base_ll, nominal_ln in levels.items():
        matched = [
            n.idx
            for n in nodes
            if math.isclose(n.kv_base_ln * math.sqrt(3.0), base_ll, rel_tol=1e-6)
        ]
        if not matched:
            raise ValueError(f"{model.network_id}: no bus on the declared {base_ll} kV level")
        for i in matched:
            nodes[i] = replace(nodes[i], kv_nominal_ln=nominal_ln)
    return ElectricalModel(model.network_id, model.variant_id, nodes, model.elements)


def prepare_network(
    network: Network,
    options: PrepareOptions | None = None,
    root: Path | None = None,
    *,
    variant_id: str | None = None,
    edits: Sequence[str] = (),
) -> PreparedNetwork:
    """Run the import pipeline on ``network`` from the cache under ``root``.

    ``edits`` (a variant's script) run right after the catalog recipe, before the load
    normalisation; the model is extracted as ``variant_id`` (default: the base variant).
    """
    opts = options or PrepareOptions()
    ctx = DssContext.new()
    compiled = compile_network(ctx, network, root)
    for command in edits:
        ctx.command(command)
    loads = normalize_loads(ctx, model=opts.load_model, vminpu=opts.vminpu, vmaxpu=opts.vmaxpu)
    coverage = coverage_report(ctx)
    ctx.set_options(tolerance=opts.tolerance, max_iterations=opts.max_iterations)
    status = ctx.solve()
    variant = variant_id or base_variant_id(network.id)
    model = extract_electrical(ctx, network.id, variant)
    as_loaded = check_bases(ctx, model)
    fix = tuple(fix_bases(ctx, as_loaded)) if as_loaded.rebase else ()
    if fix:
        status = ctx.solve()
        model = extract_electrical(ctx, network.id, variant)
    model = with_nominal(model, network.nominal_kv_ln)
    bases = check_bases(ctx, model) if fix else as_loaded
    return PreparedNetwork(
        network=network,
        options=opts,
        ctx=ctx,
        model=model,
        compiled=compiled,
        loads=loads,
        coverage=coverage,
        solve=status,
        bases_as_loaded=as_loaded,
        base_fix=fix,
        bases=bases,
        s_base=s_base_va(ctx, model),
        neutral=classify_neutral(model),
        phases=phase_consistency(model),
        setpoints=setpoint_report(ctx),
        excluded=tuple(disabled_elements(ctx)),
        edits=tuple(edits),
    )
