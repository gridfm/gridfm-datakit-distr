"""Network passports (outputs.md §3): provenance, recipe, normalisations, checks.

A passport is a JSON document per network: what was compiled and from where (licence, citation,
checksums), how (recipe and normalisations) and what came
out (counts, voltage levels, neutral representation, base case). ``inspect_text`` renders the
human summary of ``gridfm_datakit_distr inspect``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from gridfm_datakit_distr.io.schema import SCHEMA_VERSION
from gridfm_datakit_distr.model.units import SQRT3
from gridfm_datakit_distr.networks.catalog import Catalog
from gridfm_datakit_distr.networks.prepare import PreparedNetwork

SLACK_DEFINITION = (
    "Vsource bus nodes at their solved voltages; the source impedance is not part of the "
    "network admittance"
)
PHASES = ("a", "b", "c", "s1", "s2")


def _voltage_levels(prepared: PreparedNetwork) -> list[dict[str, Any]]:
    levels: dict[float, dict[str, Any]] = {}
    for zone in prepared.bases.zones:
        if zone.nominal_kv_ln is None:
            continue
        key = round(zone.nominal_kv_ln, 9)
        level = levels.setdefault(
            key, {"kv_ln": zone.nominal_kv_ln, "kv_ll": zone.nominal_kv_ln * SQRT3,
                  "zones": 0, "buses": 0},
        )  # fmt: skip
        level["zones"] += 1
        level["buses"] += len(zone.buses)
    return [levels[k] for k in sorted(levels, reverse=True)]


def _total_load(prepared: PreparedNetwork) -> dict[str, Any]:
    api = prepared.ctx.api
    p = q = 0.0
    n = 0
    with prepared.ctx.engine_call("Loads"):
        i = api.Loads.First()
        while i > 0:
            p += float(api.Loads.kW())
            q += float(api.Loads.kvar())
            n += 1
            i = api.Loads.Next()
    return {"n_loads": n, "p_kw": p, "q_kvar": q}


def _phase_voltages(prepared: PreparedNetwork) -> list[float]:
    """Min and max phase-to-ground voltage (pu of the bus base) of the base case."""
    v = np.abs(prepared.ctx.ynode_voltages())
    nodes = prepared.model.nodes
    pu = [v[n.idx] / (n.kv_base_ln * 1e3) for n in nodes if n.conductor in PHASES]
    return [float(min(pu)), float(max(pu))] if pu else []


def build_passport(prepared: PreparedNetwork, catalog: Catalog) -> dict[str, Any]:
    """The passport of a prepared network (JSON-serialisable)."""
    network = prepared.network
    source = catalog.source(network.source)
    model = prepared.model
    compiled = prepared.compiled
    normalisation: list[dict[str, Any]] = []
    if compiled.base_loads is not None:
        b = compiled.base_loads
        normalisation.append({
            "step": "base_loads", "policy": network.base_loads,
            "n_loads": b.n_loads, "total_kw_before": b.total_kw_before, "total_kw": b.total_kw,
            "total_kvar": b.total_kvar, "coincident_peak_ratio": b.coincident_peak_ratio,
            "window_h": b.window_h,
            "commands_sha256": b.commands_sha256,
        })  # fmt: skip
    loads = prepared.loads
    normalisation.append({
        "step": "loads", "model": loads.model, "vminpu": loads.vminpu,
        "vmaxpu": loads.vmaxpu, "n_loads": loads.n_loads,
        "models_before": {str(k): v for k, v in loads.models_before.items()},
    })  # fmt: skip
    normalisation.append({
        "step": "voltage_bases", "rebased_buses": len(prepared.base_fix),
        "max_deviation_as_loaded": prepared.bases_as_loaded.max_deviation,
        "max_deviation": prepared.bases.max_deviation,
        "ambiguous": list(prepared.bases.ambiguous), "conflicts": list(prepared.bases.conflicts),
    })  # fmt: skip
    known = []
    if network.extra_commands or network.base_loads != "as_is":
        known.append("source model compiled with corrections (recipe: extra_commands, base_loads), "
                     "each explained in the package's networks/catalog.yaml")  # fmt: skip
    if prepared.base_fix:
        known.append(f"{len(prepared.base_fix)} buses re-based from OpenDSS's nearest listed base "
                     "to their zone nominal voltage")  # fmt: skip
    zones = Counter(z.kind for z in prepared.neutral.zones)
    by_class = Counter(e.element_class for e in model.elements)
    neutral_ok = (
        network.neutral in (None, prepared.neutral.representation)
        and not prepared.neutral.floating_islands
        and (prepared.neutral.representation != "mixed" or network.neutral == "mixed")
    )
    return {
        "network_id": network.id,
        "schema_version": SCHEMA_VERSION,
        "source": source.id,
        "source_url": source.homepage,
        "doi": source.doi,
        "license": source.license,
        "citation": source.citation,
        "country": source.country,
        "checksums": {
            "source_manifest_sha256": source.manifest_sha256,
            "master_sha256": source.manifest.get(network.master),
            "compiled_sha256": compiled.master_sha256,
        },
        "original_format": "opendss",
        "converter": None,
        "engine_version": prepared.ctx.engine_version(),
        "recipe": {
            "master": network.master,
            "master_stop_before": network.master_stop_before,
            "extra_commands": list(compiled.extra_commands),
            "base_loads": network.base_loads,
            "load_actions": list(compiled.load_actions),
        },
        "normalisation": normalisation,
        "s_base_va": prepared.s_base[0],
        "s_base_origin": prepared.s_base[1],
        "frequency_hz": float(prepared.ctx.api.Solution.Frequency()),
        "neutral_representation": prepared.neutral.representation,
        "neutral": {
            "expected": network.neutral,
            "zones": dict(sorted(zones.items())),
            "grounded_neutrals": len(prepared.neutral.grounded_neutrals),
            "floating_islands": len(prepared.neutral.floating_islands),
            "grounded_pc_in_explicit": len(prepared.neutral.grounded_pc_in_explicit),
        },
        "voltage_levels": _voltage_levels(prepared),
        "counts": {
            "buses": len({n.bus for n in model.nodes}),
            "nodes": model.n_nodes,
            "elements": len(model.elements),
            "by_class": dict(sorted(by_class.items())),
            "disabled": len(prepared.excluded),
        },
        "has_split_phase": any(n.conductor in ("s1", "s2") for n in model.nodes),
        "total_load": _total_load(prepared),
        "base_case": {
            "controls": "ControlMode OFF, taps and capacitor states as compiled",
            "converged": prepared.solve.converged,
            "iterations": prepared.solve.iterations,
            "tolerance": prepared.options.tolerance,
            "setpoints_max_rel": prepared.setpoints.max_rel,
            "off_setpoint": len(prepared.setpoints.off),
            "phase_voltage_pu": _phase_voltages(prepared),
        },
        "checks": {
            "C1_converged": prepared.solve.converged,
            "C4_coverage": prepared.coverage.passes(),
            "C5_phases": prepared.phases.passes(),
            "C6_neutral": neutral_ok,
            "bases": prepared.bases.passes(),
            "setpoints": prepared.setpoints.passes(),
        },
        "slack_definition": SLACK_DEFINITION,
        "known_differences": known,
        "excluded_elements": list(prepared.excluded),
    }


def write_passport(passport: dict[str, Any], directory: Path) -> Path:
    """Write ``<directory>/<network_id>.json`` (ids with '/' make sub-folders)."""
    path = directory / f"{passport['network_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(passport, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def inspect_text(passport: dict[str, Any]) -> str:
    """Human summary of a passport (``gridfm_datakit_distr inspect``)."""
    p = passport
    counts, recipe, base = p["counts"], p["recipe"], p["base_case"]
    classes = ", ".join(f"{k} {v}" for k, v in counts["by_class"].items())
    levels = ", ".join(
        f"{lv['kv_ll']:.4g}" + (f" (x{lv['zones']})" if lv["zones"] > 1 else "")
        for lv in p["voltage_levels"]
    )
    zones = ", ".join(f"{v} {k}" for k, v in p["neutral"]["zones"].items())
    load = p["total_load"]
    lines = [
        f"{p['network_id']} ({p['source']}; licence: {p['license']})",
        f"master: {recipe['master']}"
        + (f" (up to '{recipe['master_stop_before']}')" if recipe["master_stop_before"] else ""),
        f"recipe: {len(recipe['extra_commands'])} extra commands, "
        f"base loads {recipe['base_loads']}",
        f"buses {counts['buses']}, nodes {counts['nodes']}, elements {counts['elements']}"
        f" (+{counts['disabled']} disabled)",
        f"elements: {classes}",
        f"neutral: {p['neutral_representation']} (zones: {zones}); grounded neutrals "
        f"{p['neutral']['grounded_neutrals']}, floating islands {p['neutral']['floating_islands']}",
        f"split phase: {'yes' if p['has_split_phase'] else 'no'}",
        f"voltage levels (kV LL): {levels}",
        f"s_base: {p['s_base_va'] / 1e3:.6g} kVA ({p['s_base_origin']})",
        f"total load: {load['p_kw']:.6g} kW, {load['q_kvar']:.6g} kvar ({load['n_loads']} loads)",
        "normalisation:",
    ]
    for step in p["normalisation"]:
        if step["step"] == "loads":
            before = ", ".join(f"model {k}: {v}" for k, v in step["models_before"].items())
            window = f"vminpu {step['vminpu']}, vmaxpu {step['vmaxpu']}"
            lines.append(
                f"  loads: {step['model']}, {window} ({step['n_loads']} loads; before: {before})"
            )
        elif step["step"] == "voltage_bases":
            lines.append(f"  voltage bases: {step['rebased_buses']} buses re-based; max deviation "
                         f"{step['max_deviation_as_loaded']:.1e} as loaded, "
                         f"{step['max_deviation']:.1e} after")  # fmt: skip
        elif step["step"] == "base_loads":
            ratio = step["coincident_peak_ratio"]
            change = (f"{step['total_kw_before']:.6g} -> {step['total_kw']:.6g} kW, coincident "
                      f"peak / mean {ratio:.6g} over {step['window_h']:g} h")  # fmt: skip
            lines.append(f"  base loads: {step['policy']} ({step['n_loads']} loads, {change})")
    vmin, vmax = base["phase_voltage_pu"] or [float("nan")] * 2
    lines.append(
        f"base case (taps as compiled): {'converged' if base['converged'] else 'NOT converged'} in "
        f"{base['iterations']} iterations (Tolerance {base['tolerance']:g}); setpoints max "
        f"deviation {base['setpoints_max_rel']:.0e}, {base['off_setpoint']} off; phase voltages "
        f"{vmin:.4f}-{vmax:.4f} pu"
    )
    failed = [name for name, ok in p["checks"].items() if not ok]
    lines.append("checks: " + ("all pass" if not failed else "FAIL " + ", ".join(failed)))
    return "\n".join(lines) + "\n"
