"""The voltage source of a model: ideal voltages behind an admittance, read from the engine.

:class:`~gridfm_datakit_distr.model.electrical.ElectricalModel` keeps neither the Vsource's Yprim
(it is not a power-delivery element) nor its base voltage and angle; the OPF needs them, and the
dataset stores them (``variants/source``, ``variants/source_yprim``, :func:`source_tables`) so
that a reader can rebuild the source without OpenDSS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element

if TYPE_CHECKING:
    from gridfm_datakit_distr.engine.context import DssContext


@dataclass(frozen=True, slots=True, eq=False)
class Source:
    """The voltage source: ideal voltages behind an admittance, at its terminal's nodes."""

    nodes: NDArray[np.int64]
    """Model nodes of the source terminal, in conductor order."""
    y_s: NDArray[np.complex128]
    """Admittance (S) between the ideal source and :attr:`nodes`."""
    e_v: NDArray[np.complex128]
    """Ideal source voltages (V) at 1 pu; a scenario's are ``source_v_pu * e_v``."""


def read_source(ctx: DssContext, model: ElectricalModel) -> Source:
    """The source of ``model`` from its compiled circuit in ``ctx``.

    :class:`ElectricalModel` keeps neither the Vsource's Yprim (it is not a PD element) nor its
    base voltage and angle, so they are read from the engine: ``Y_s`` is the Yprim block of the
    first terminal (the second is ground, :func:`unsupported_reason`), ``E`` has the magnitude
    ``BasekV`` (line-to-neutral) and the angles ``AngleDeg - 120°·k`` per conductor ``k``.
    Raises ``ValueError`` unless the model has one enabled Vsource, grounded at its second
    terminal.
    """
    vs = vsource_of(model)
    keep = [k for k, node in enumerate(vs.terminals[0]) if node != GROUND]
    y_s = ctx.element_yprim(vs.name)[np.ix_(keep, keep)]
    api = ctx.api
    with ctx.engine_call(f"Vsources {vs.name}"):
        api.Vsources.Name(vs.name.split(".", 1)[1])
        base_kv, angle_deg = float(api.Vsources.BasekV()), float(api.Vsources.AngleDeg())
        phases = int(api.Vsources.Phases())
    v_ln = base_kv * 1e3 / (np.sqrt(3.0) if phases > 1 else 1.0)
    angles = np.deg2rad(angle_deg) - 2.0 * np.pi / 3.0 * np.array(keep)
    nodes = np.array([vs.terminals[0][k] for k in keep], dtype=np.int64)
    return Source(nodes=nodes, y_s=y_s, e_v=v_ln * np.exp(1j * angles))


def vsource_of(model: ElectricalModel) -> Element:
    """The model's one enabled Vsource, grounded at its second terminal (``ValueError`` if not)."""
    found = [e for e in model.elements if e.enabled and e.element_class == "vsource"]
    if len(found) != 1:
        raise ValueError(f"{model.variant_id}: {len(found)} enabled voltage sources, one expected")
    if any(n != GROUND for terminal in found[0].terminals[1:] for n in terminal):
        raise ValueError(f"{found[0].name} is not grounded at its second terminal")
    return found[0]


def source_tables(model: ElectricalModel, source: Source) -> dict[str, pa.Table]:
    """``variants/source`` and ``variants/source_yprim`` rows of ``source`` (outputs.md §4)."""
    vs = vsource_of(model)
    positions = np.array([vs.terminals[0].index(int(n)) for n in source.nodes], dtype=np.int8)
    n = positions.size
    rows, cols = np.nonzero(source.y_s)
    vid = model.variant_id
    return {
        "variants/source": pa.table(
            {
                "variant_id": [vid] * n,
                "element_idx": np.full(n, vs.idx, dtype=np.int32),
                "position": positions,
                "node_idx": source.nodes.astype(np.int32),
                "e_re": source.e_v.real,
                "e_im": source.e_v.imag,
            },
            schema=schema.SOURCE,
        ),
        "variants/source_yprim": pa.table(
            {
                "variant_id": [vid] * rows.size,
                "row": positions[rows].astype(np.int16),
                "col": positions[cols].astype(np.int16),
                "g": source.y_s[rows, cols].real,
                "b": source.y_s[rows, cols].imag,
            },
            schema=schema.SOURCE_YPRIM,
        ),
    }
