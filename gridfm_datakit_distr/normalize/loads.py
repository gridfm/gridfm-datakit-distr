"""Load normalisation.

Every enabled load gets the constant-power model (``model=1``) and the voltage window
[``vminpu``, ``vmaxpu``] (defaults 0.6 and 1.5) through OpenDSS ``Edit`` commands; connection,
phases and setpoints are not touched. With ``model="as_is"`` only the window is set. Outside the
window OpenDSS silently changes the load model and still reports convergence, so
every solve is checked against the setpoints (``validate.setpoints``).

Generators of the source models are not edited (none of the catalog networks has one); the
setpoint check covers them.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from gridfm_datakit_distr.engine.context import DssContext

LoadModel = Literal["constant_power", "as_is"]


class LoadNormalisationError(RuntimeError):
    """An edit did not take effect."""


@dataclass(frozen=True, slots=True)
class LoadNormalisation:
    """What :func:`normalize_loads` did, for the passport."""

    model: LoadModel
    vminpu: float
    vmaxpu: float
    n_loads: int
    models_before: dict[int, int]
    """OpenDSS load model number -> number of enabled loads that had it."""
    commands: tuple[str, ...]


def _enabled_loads(ctx: DssContext) -> list[tuple[str, int]]:
    """(name, model) of the enabled loads; ``Loads.First/Next`` skips disabled ones."""
    api = ctx.api
    loads: list[tuple[str, int]] = []
    with ctx.engine_call("Loads"):
        i = api.Loads.First()
        while i > 0:
            loads.append((str(api.Loads.Name()), int(api.Loads.Model())))
            i = api.Loads.Next()
    return loads


def normalize_loads(
    ctx: DssContext,
    *,
    model: LoadModel = "constant_power",
    vminpu: float = 0.6,
    vmaxpu: float = 1.5,
) -> LoadNormalisation:
    """Apply the load normalisation to the compiled circuit of ``ctx`` and read it back."""
    if not 0.0 < vminpu < 1.0 < vmaxpu:
        raise ValueError(f"need 0 < vminpu < 1 < vmaxpu, got {vminpu}, {vmaxpu}")
    if model not in ("constant_power", "as_is"):
        raise ValueError(f"model must be constant_power or as_is, got {model!r}")
    loads = _enabled_loads(ctx)
    set_model = " model=1" if model == "constant_power" else ""
    commands = [
        f"Edit Load.{name}{set_model} vminpu={vminpu!r} vmaxpu={vmaxpu!r}" for name, _ in loads
    ]
    api = ctx.api
    for command, (name, before) in zip(commands, loads, strict=True):
        ctx.command(command)
        with ctx.engine_call(f"Loads {name}"):
            api.Loads.Name(name)
            got = (int(api.Loads.Model()), float(api.Loads.Vminpu()), float(api.Loads.Vmaxpu()))
        want = (1 if model == "constant_power" else before, vminpu, vmaxpu)
        if got != want:  # an Edit of a missing element is silent
            raise LoadNormalisationError(f"{command}: Load.{name} reads {got}, expected {want}")
    return LoadNormalisation(
        model=model,
        vminpu=vminpu,
        vmaxpu=vmaxpu,
        n_loads=len(loads),
        models_before=dict(sorted(Counter(before for _, before in loads).items())),
        commands=tuple(commands),
    )


@dataclass(frozen=True, slots=True)
class BaseLoads:
    """What :func:`loads_from_loadshape_means` did, for the passport."""

    n_loads: int
    total_kw_before: float
    total_kw: float
    total_kvar: float
    commands_sha256: str
    """SHA-256 of the newline-joined ``Edit`` commands (one per load)."""


def loads_from_loadshape_means(ctx: DssContext) -> BaseLoads:
    """Set each enabled load's base kW and kvar to the mean of its loadshape.

    The daily loadshape, else the yearly one. With ``UseActual`` the multipliers are kW and kvar;
    otherwise they scale the load's base values. Without a Q curve the load keeps its kvar/kW
    ratio, as OpenDSS does along the curve. A load without a loadshape raises: the source gives
    no operating point for it.
    """
    api = ctx.api
    loads: list[tuple[str, float, float, str]] = []
    with ctx.engine_call("Loads"):
        i = api.Loads.First()
        while i > 0:
            shape = str(api.Loads.Daily()) or str(api.Loads.Yearly())
            loads.append((str(api.Loads.Name()), float(api.Loads.kW()), float(api.Loads.kvar()),
                          shape))  # fmt: skip
            i = api.Loads.Next()
    commands = []
    total_kw = total_kvar = 0.0
    for name, kw, kvar, shape in loads:
        if not shape:
            raise LoadNormalisationError(f"Load.{name} has no daily or yearly loadshape")
        with ctx.engine_call(f"LoadShape {shape}"):
            api.LoadShape.Name(shape)
            p = np.asarray(api.LoadShape.PMult(), dtype=float)
            q = np.asarray(api.LoadShape.QMult(), dtype=float)
            actual = bool(api.LoadShape.UseActual())
        # QMult() reads [0.0] when the shape has no Q curve; the property text is empty.
        has_q = bool(ctx.command(f"? LoadShape.{shape}.qmult").strip())
        if p.size == 0:
            raise LoadNormalisationError(f"LoadShape.{shape} of Load.{name} has no points")
        kw_mean = float(p.mean()) * (1.0 if actual else kw)
        if has_q:
            kvar_mean = float(q.mean()) * (1.0 if actual else kvar)
        else:
            kvar_mean = kw_mean * kvar / kw if kw else 0.0
        command = f"Edit Load.{name} kW={kw_mean!r} kvar={kvar_mean!r}"
        ctx.command(command)
        with ctx.engine_call(f"Loads {name}"):
            api.Loads.Name(name)
            got = (float(api.Loads.kW()), float(api.Loads.kvar()))
        if not np.allclose(got, (kw_mean, kvar_mean), rtol=1e-12, atol=1e-12):
            raise LoadNormalisationError(f"{command}: Load.{name} reads {got}")
        commands.append(command)
        total_kw += kw_mean
        total_kvar += kvar_mean
    digest = hashlib.sha256("\n".join(commands).encode()).hexdigest()
    return BaseLoads(
        n_loads=len(loads),
        total_kw_before=sum(kw for _, kw, _, _ in loads),
        total_kw=total_kw,
        total_kvar=total_kvar,
        commands_sha256=digest,
    )
