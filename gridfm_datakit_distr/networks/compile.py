"""Compile a catalog network in an engine context, with its documented recipe.

The recipe of a network is part of the catalog: an optional cut of the master
(``master_stop_before``), OpenDSS commands run after it (``extra_commands``) and the base loads
policy (``base_loads``). Everything is read by the engine; the cut only stops the master
at an exact line, before the commands that run simulations and write files.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.networks.cache import master_path
from gridfm_datakit_distr.networks.catalog import CatalogError, Network
from gridfm_datakit_distr.normalize.loads import BaseLoads, loads_from_loadshape_means

PREFIX_SUFFIX = ".prefix.dss"
"""Suffix of the cut master, written next to the master so that relative redirects resolve."""


@dataclass(frozen=True, slots=True)
class CompileRecord:
    """How a network was compiled, for the passport."""

    network_id: str
    master: Path
    """The file given to ``Redirect``: the master or its cut."""
    master_sha256: str
    extra_commands: tuple[str, ...]
    load_actions: tuple[str, ...]
    """Actions of ``DssContext.load`` (``CalcVoltageBases`` if some bus had no base)."""
    base_loads: BaseLoads | None


def master_prefix(master: Path, stop_before: str) -> Path:
    """Write (atomically, if missing or different) and return the lines of ``master`` before the
    line ``stop_before``, which must occur exactly once (compared stripped, ignoring case)."""
    lines = master.read_bytes().splitlines(keepends=True)
    marker = stop_before.strip().lower()
    hits = [i for i, line in enumerate(lines) if line.decode("latin-1").strip().lower() == marker]
    if len(hits) != 1:
        raise CatalogError(f"{master}: line {stop_before!r} occurs {len(hits)} times, expected 1")
    data = b"".join(lines[: hits[0]])
    path = master.with_name(master.name + PREFIX_SUFFIX)
    if not path.is_file() or path.read_bytes() != data:
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return path


def compile_network(ctx: DssContext, network: Network, root: Path | None = None) -> CompileRecord:
    """Compile ``network`` from the cache into ``ctx`` following its catalog recipe."""
    master = master_path(network, root)
    if network.master_stop_before is not None:
        master = master_prefix(master, network.master_stop_before)
    ctx.load(master, network.extra_commands)
    base_loads = loads_from_loadshape_means(ctx) if network.base_loads == "loadshape_mean" else None
    return CompileRecord(
        network_id=network.id,
        master=master,
        master_sha256=hashlib.sha256(master.read_bytes()).hexdigest(),
        extra_commands=tuple(network.extra_commands),
        load_actions=tuple(ctx.load_actions),
        base_loads=base_loads,
    )
