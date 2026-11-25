"""Variant generator.

A variant is the network as compiled by its catalog recipe plus an OpenDSS edit script drawn from
the config. Each attempt is compiled afresh, its edits are checked against the perturbation
(:func:`check_edits` and read-backs: a wrong edit is a bug, raised), and its base case must pass
C1 (converges) and C2 (KCL residual). A variant failing C1 or C2 is redrawn with the next
attempt's streams, at most :data:`MAX_ATTEMPTS` times; then it is recorded as failed, with the
reasons, for the manifest. Datasets hold only sampled variants (``v0000`` ...); the default
config perturbs every one of them.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from gridfm_datakit_distr.config.schema import PV, PVOrientation, Storage, Variants
from gridfm_datakit_distr.config.value_spec import ValueSpec
from gridfm_datakit_distr.extract.injections import injections
from gridfm_datakit_distr.extract.source import read_source
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel
from gridfm_datakit_distr.model.ids import variant_id
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, PrepareOptions, prepare_network
from gridfm_datakit_distr.scenarios.seeds import network_key, variant_seed
from gridfm_datakit_distr.validate.residual import residual
from gridfm_datakit_distr.variants.perturbation import (
    Perturbation,
    check_edits,
    edit_script,
    sample_perturbation,
)
from gridfm_datakit_distr.variants.targets import VariantTargets
from gridfm_datakit_distr.variants.upstream import Upstream, sequence_admittance

MAX_ATTEMPTS = 3
EDIT_TOL = 1e-12
"""Largest relative deviation of an edited line's Yprim from its perturbation (round-off ~1e-15)."""


class VariantError(RuntimeError):
    """An edit script did not realise its perturbation: a bug, never resampled."""


@dataclass(frozen=True, slots=True)
class VariantResult:
    network_id: str
    variant_idx: int
    seed: int
    """Fingerprint of the variant's seed sequence (``seed`` column)."""
    status: Literal["ok", "failed"]
    perturbation: Perturbation | None
    """The accepted attempt's perturbation; None if every attempt failed."""
    script: tuple[str, ...]
    failures: tuple[str, ...]
    """``attempt <a>: <check>: <reason>`` of every rejected attempt."""
    model: ElectricalModel | None
    edit_deviation: float
    prepared: PreparedNetwork | None = field(default=None, repr=False, compare=False)
    """The accepted attempt's compiled network, kept when asked for (``keep_prepared``) for a
    :class:`~gridfm_datakit_distr.solve.pf_batch.VariantRunner` to take over."""

    @property
    def variant_id(self) -> str:
        return variant_id(self.network_id, self.variant_idx)

    @property
    def script_sha1(self) -> str:
        return hashlib.sha1("\n".join(self.script).encode("utf-8")).hexdigest()

    def record(self) -> dict[str, Any]:
        """The row of ``variants/variants`` (outputs.md §4)."""
        if self.model is None or self.perturbation is None:
            raise ValueError(f"{self.variant_id} failed: it has no record, only failures")
        return {
            "variant_id": self.variant_id,
            "network_id": self.network_id,
            "variant_idx": self.variant_idx,
            "seed": self.seed,
            "perturbation": self.perturbation.to_json(),
            "perturbation_script_sha1": self.script_sha1,
            "n_nodes": len(self.model.nodes),
            "n_elements": len(self.model.elements),
        }


def physics_failures(prepared: PreparedNetwork) -> list[str]:
    """C1 and C2 on the base case of a prepared variant; empty when both pass."""
    if not prepared.solve.converged:
        return [f"C1: not converged after {prepared.solve.iterations} iterations"]
    ctx, model = prepared.ctx, prepared.model
    r = residual(
        model, ctx.ynode_voltages(), injections(ctx, model), tolerance=prepared.options.tolerance
    )
    return [] if r.passes() else [f"C2: KCL residual ratio {r.ratio:.1e} at {r.worst_node}"]


def readback_mismatches(p: Perturbation, prepared: PreparedNetwork) -> list[str]:
    """Grounding resistances, load connections, DER units and the substation (source
    admittance, lumped loads) of ``prepared`` that differ from ``p``."""
    api, model = prepared.ctx.api, prepared.model
    out = []
    for key, ohm in p.grounding_ohm.items():
        element, _, winding = key.partition("/w")
        cls, name = element.split(".", 1)
        if cls == "Reactor":
            api.Reactors.Name(name)
            got = float(api.Reactors.R())
        else:
            api.Transformers.Name(name)
            api.Transformers.Wdg(int(winding))
            got = float(api.Transformers.Rneut())
        if got != ohm:
            out.append(f"{key}: resistance {got!r}, drawn {ohm!r}")
    elements = {e.name: e for e in model.elements}
    for load, (_, new) in p.phases.items():
        bus, *conductors = new.split(".")
        nodes = [
            model.nodes[k].opendss_node if k != GROUND else "0" for k in elements[load].terminals[0]
        ]
        expected = [f"{bus}.{c}" for c in conductors] + ["0"] * (len(nodes) - len(conductors))
        if nodes != expected:
            out.append(f"{load}: nodes {nodes}, drawn {expected}")

    def nodes_of(name: str) -> list[int] | None:
        e = elements.get(name)
        return None if e is None else e.terminals[0]

    units = [("pv", load) for load in p.pv_kwp] + [("bess", load) for load in p.storage_kw]
    units += [("dg", load) for load in p.dg_kw]
    for prefix, load in units:
        generator = f"Generator.{prefix}_{load.split('.', 1)[1]}"
        if nodes_of(generator) is None or nodes_of(generator) != nodes_of(load):
            out.append(f"{generator}: not on the nodes of {load}")
    if p.upstream is not None:
        out += _substation_mismatches(p, prepared)
    return out


def _substation_mismatches(p: Perturbation, prepared: PreparedNetwork) -> list[str]:
    """The source's admittance against the one its sequence impedances give, and the lumped
    loads on their nodes."""
    up = p.upstream
    assert up is not None
    out = []
    y = sequence_admittance(up.z1_ohm, up.z0_ohm)
    y_s = read_source(prepared.ctx, prepared.model).y_s
    deviation = float(np.abs(y_s - y).max() / np.abs(y).max())
    if deviation > EDIT_TOL:
        out.append(f"{up.site.vsource}: admittance deviates by {deviation:.1e} from Z1, Z0")
    model = prepared.model
    elements = {e.name: e for e in model.elements}
    for k, load in enumerate(up.site.loads):
        e = elements.get(load)
        nodes = None if e is None else [
            model.nodes[n].opendss_node if n != GROUND else "0" for n in e.terminals[0]
        ]  # fmt: skip
        bus, *conductors = up.site.bus1(k).split(".")
        expected = [f"{bus}.{c}" for c in conductors] + (["0"] if len(conductors) == 1 else [])
        if nodes != expected:
            out.append(f"{load}: nodes {nodes}, expected {expected}")
    return out


def generate_variant(
    network: Network,
    targets: VariantTargets,
    config: Variants,
    seed: int,
    k: int,
    *,
    options: PrepareOptions | None = None,
    root: Path | None = None,
    der: PV | None = None,
    storage: Storage | None = None,
    dg_kw: float | None = None,
    upstream: Upstream | None = None,
    load_level: ValueSpec | None = None,
    orientation: PVOrientation | None = None,
    keep_prepared: bool = False,
) -> VariantResult:
    """Variant ``k`` of ``network``; with ``der``, the PV units of its customers, with
    ``storage`` the storage units of those with PV, with ``dg_kw`` the network's dispatchable
    generators at that capacity (kW), with ``upstream`` its substation, with ``load_level`` the
    spec of its load level (:func:`sample_perturbation`), with ``orientation`` its PV units'
    orientation classes. Its draws depend on the network's id, not on its place in the dataset.
    With ``keep_prepared`` the result carries the accepted attempt's compiled network (its
    OpenDSS context stays alive)."""
    vid = variant_id(network.id, k)
    net = network_key(network.id)
    fingerprint = variant_seed(seed, net, k)
    opts = options or PrepareOptions()
    failures: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        p = sample_perturbation(
            config, targets, seed, net, k, attempt, der=der, storage=storage, dg_kw=dg_kw,
            upstream=upstream, load_level=load_level, orientation=orientation,
        )  # fmt: skip
        script = edit_script(p, targets, (opts.vminpu, opts.vmaxpu))
        prepared = prepare_network(network, options, root, variant_id=vid, edits=script)
        deviation = check_edits(p, targets, prepared.model)
        if deviation > EDIT_TOL:
            raise VariantError(
                f"{vid}: edits not realised: line Yprims deviate from the perturbation by "
                f"{deviation:.1e} (tolerance {EDIT_TOL:g})"
            )
        mismatches = readback_mismatches(p, prepared)
        if mismatches:
            raise VariantError(f"{vid}: edits not realised: {'; '.join(mismatches[:3])}")
        reasons = physics_failures(prepared)
        if not reasons:
            return VariantResult(
                network.id, k, fingerprint, "ok", p, tuple(script), tuple(failures),
                prepared.model, deviation, prepared if keep_prepared else None,
            )  # fmt: skip
        failures += [f"attempt {attempt}: {reason}" for reason in reasons]
    return VariantResult(
        network.id, k, fingerprint, "failed", None, (), tuple(failures), None, math.nan
    )


def variants_table(results: list[VariantResult]) -> pa.Table:
    """``variants/variants`` rows of the accepted variants (outputs.md §4)."""
    rows = [r.record() for r in results if r.status == "ok"]
    return pa.Table.from_pylist(rows, schema=schema.VARIANTS)
