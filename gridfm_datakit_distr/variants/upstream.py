"""The MV/LV substation behind the source of an LV feeder (``variants.upstream``).

The CSIRO feeders (``csiro_enwl/nNN_fM``) are supplied by an almost ideal LV source, without
the MV/LV transformer, the MV network or the other feeders of their substation. With
``variants.upstream`` each variant draws its substation:

- the transformer rating ``S_r`` (kVA) from :data:`RATING_COUNTS`, the ratings of ENWL's
  single-transformer substations whose customer count lies in the same octave as the network's
  (its customers: the loads of all its feeders);
- its short-circuit impedance ``z_k``: the minimum of IEC 60076-5:2006 Table 1 for ``S_r``
  (:data:`IEC_60076_5_MIN_Z`); its resistance ``P_k / S_r`` from the Tier 2 maximum load losses of
  Regulation (EU) 548/2014 Annex I Table I.1 (:data:`EU_548_TIER2_PK_W`, linear between rows as
  the Regulation's footnote says); its reactance ``sqrt(z_k² - r²)``;
- the MV network: :data:`MV_FAULT_VA` at the MV bus with the transformer's X/R, in the positive
  and negative sequences only (a Dyn transformer blocks the upstream zero sequence).

The Vsource gets ``Z1 = Z2 = Z_T + Z_MV`` and ``Z0 = Z_T`` (ohm at its base voltage): with the
feeder's neutral bonded at the source this is the Thevenin equivalent of a Dyn transformer at
every node (spike s69). No node, element or base is added for it; the slack stays the source.
The other feeders of the substation become three single-phase constant-power loads at the source
bus, ``Load.upstream_feeders_<a|b|c>``, each the scenario's global load factor times the sum of
the nominal loads on that phase of the sibling feeders (read through the engine). They are not
customers: no PV, storage, DG, phase move or per-customer factor.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.source import vsource_of
from gridfm_datakit_distr.model.electrical import GROUND
from gridfm_datakit_distr.networks.catalog import Catalog, Network
from gridfm_datakit_distr.networks.compile import compile_network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork

RATING_COUNTS: dict[int, dict[float, int]] = {
    4: {25: 479, 50: 854, 100: 402, 150: 1, 200: 133, 300: 14, 315: 59, 500: 107, 750: 11,
        800: 73, 1000: 82, 1500: 1},
    8: {25: 137, 50: 800, 100: 650, 200: 266, 300: 24, 315: 115, 500: 180, 750: 23, 800: 116,
        1000: 154, 1500: 1},
    16: {25: 4, 50: 228, 100: 585, 150: 1, 200: 272, 300: 34, 315: 175, 500: 221, 600: 1,
         750: 37, 800: 112, 1000: 117},
    32: {50: 23, 100: 307, 150: 1, 200: 294, 250: 1, 300: 114, 315: 324, 350: 1, 400: 1,
         500: 387, 600: 1, 750: 53, 800: 127, 1000: 136, 1400: 1, 1500: 1},
    64: {50: 2, 100: 44, 200: 179, 250: 1, 300: 267, 315: 618, 400: 2, 500: 807, 750: 140,
         800: 226, 1000: 213, 1500: 3},
    128: {100: 1, 200: 24, 250: 4, 300: 315, 315: 695, 400: 5, 500: 2022, 600: 2, 750: 490,
          800: 379, 1000: 329, 1250: 1},
    256: {300: 35, 315: 76, 400: 1, 500: 1164, 600: 4, 750: 489, 800: 569, 1000: 475, 1200: 1,
          1500: 4},
    512: {315: 1, 500: 16, 750: 52, 800: 76, 1000: 156, 1500: 4},
}  # fmt: skip
"""Distribution transformers per rating (kVA), by the customer count of their substation:
key ``c`` holds the substations with ``c <= customers < 2c`` (the last, 512, has no upper
bound). Derived numbers (counts) from Electricity North West's open data, licensed under CC BY
4.0: the distribution-transformer capacity register (``sp-enw-capacity-distribution-tx``)
joined on the substation number with the smart-meter installation counts
(``smart-meter-installation``, ``TOTAL CUSTOMERS``); single-transformer substations only,
ratings of 25 kVA and above; 18 133 substations."""

IEC_60076_5_MIN_Z = ((630.0, 0.04), (1250.0, 0.05), (2500.0, 0.06), (6300.0, 0.07))
"""Minimum short-circuit impedance (pu) of two-winding transformers up to each rating (kVA),
IEC 60076-5:2006 Table 1 (from 25 kVA)."""

EU_548_TIER2_PK_W = (
    (25.0, 600.0), (50.0, 750.0), (100.0, 1250.0), (160.0, 1750.0), (250.0, 2350.0),
    (315.0, 2800.0), (400.0, 3250.0), (500.0, 3900.0), (630.0, 4600.0), (800.0, 6000.0),
    (1000.0, 7600.0), (1250.0, 9500.0), (1600.0, 12000.0), (2000.0, 15000.0),
    (2500.0, 18500.0), (3150.0, 23000.0),
)  # fmt: skip
"""Maximum load losses (W) of three-phase liquid-immersed transformers (Um <= 24 kV / <= 3.6 kV)
at each rating (kVA), Tier 2 (from 1 July 2021): Regulation (EU) 548/2014 Annex I Table I.1."""

MV_FAULT_VA = 250e6
"""Three-phase fault level of the MV network (VA): ENWL's design fault level at 11 kV."""

LOAD_PREFIX = "upstream_feeders_"
PHASES = ("a", "b", "c")
_ID = re.compile(r"csiro_enwl/n(?P<net>\d+)_f(?P<feeder>\d+)")


@dataclass(frozen=True, slots=True)
class Upstream:
    """The substation of a network: its other feeders and where the source sits, read once per
    run from the engine."""

    network_id: str
    siblings: tuple[str, ...]
    """The other feeders of the substation (network ids)."""
    customers: int
    """Enabled loads of all its feeders, this one included."""
    vsource: str
    """The source, ``Vsource.<name>``."""
    phase_nodes: tuple[str, ...]
    """OpenDSS nodes of the source's conductors, e.g. ``sourcebus.1``."""
    neutral: str | None
    """Conductor number of the source bus's neutral (``4``), None without one."""
    kv: float
    """Rated voltage of the feeder's loads (kV), that of the lumped loads."""
    kv_ll: float
    """Base voltage of the source, line to line (kV)."""
    p_w: tuple[float, ...]
    """Nominal active power of the other feeders per source conductor (W)."""
    q_var: tuple[float, ...]
    """Nominal reactive power of the other feeders per source conductor (var)."""

    @property
    def loads(self) -> tuple[str, ...]:
        """The lumped loads, ``Load.upstream_feeders_<a|b|c>``, one per source conductor."""
        return tuple(f"Load.{LOAD_PREFIX}{PHASES[k]}" for k in range(len(self.phase_nodes)))

    def bus1(self, k: int) -> str:
        """``bus1`` of the lumped load of source conductor ``k`` (to the neutral, else ground)."""
        node = self.phase_nodes[k]
        return node if self.neutral is None else f"{node}.{self.neutral}"

    def record(self) -> dict[str, Any]:
        """The network's entry in ``manifest.json`` (``networks[].upstream``)."""
        return {
            "siblings": list(self.siblings),
            "customers": self.customers,
            "p_w": list(self.p_w),
            "q_var": list(self.q_var),
        }


def in_scope(network_id: str) -> bool:
    """Whether ``variants.upstream`` applies to the network: a CSIRO feeder."""
    return _ID.fullmatch(network_id) is not None


def rating_pmf(customers: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ratings (kVA, ascending) and their probabilities for a substation of ``customers``."""
    edges = sorted(RATING_COUNTS)
    if customers < edges[0]:
        raise ValueError(f"{customers} customers: the rating table starts at {edges[0]}")
    counts = RATING_COUNTS[edges[bisect.bisect_right(edges, customers) - 1]]
    ratings = np.array(sorted(counts), dtype=np.float64)
    n = np.array([counts[r] for r in sorted(counts)], dtype=np.float64)
    return ratings, n / n.sum()


def draw_rating(customers: int, rng: np.random.Generator) -> float:
    """A rating (kVA) for a substation of ``customers``: one uniform draw, inverse CDF."""
    ratings, p = rating_pmf(customers)
    j = int(np.searchsorted(np.cumsum(p), rng.random(), side="right"))
    return float(ratings[min(j, ratings.size - 1)])


def transformer_impedance(kva: float) -> tuple[float, float]:
    """Resistance and reactance (pu of the rating) of a transformer of ``kva``: ``z_k`` of IEC
    60076-5 Table 1, ``r = P_k / S_r`` with ``P_k`` of Regulation 548/2014 Table I.1 Tier 2
    (linear between its rows), ``x = sqrt(z_k² - r²)``. ``ValueError`` outside 25-3150 kVA."""
    sizes = [s for s, _ in EU_548_TIER2_PK_W]
    if not sizes[0] <= kva <= sizes[-1]:
        raise ValueError(f"{kva} kVA is outside the tables (25 to 3150 kVA)")
    z = next(z for top, z in IEC_60076_5_MIN_Z if kva <= top)
    pk = float(np.interp(kva, sizes, [p for _, p in EU_548_TIER2_PK_W]))
    r = pk / (kva * 1e3)
    return r, math.sqrt(z * z - r * r)


def source_impedance(kva: float, kv_ll: float) -> tuple[complex, complex]:
    """``Z1`` (= ``Z2``) and ``Z0`` (ohm at ``kv_ll``) of the substation of a ``kva``
    transformer: the transformer plus the MV network in the positive sequence, the transformer
    alone in the zero sequence."""
    v2 = (kv_ll * 1e3) ** 2
    r, x = transformer_impedance(kva)
    zt = complex(r, x) * v2 / (kva * 1e3)
    zmv = v2 / MV_FAULT_VA * complex(r, x) / abs(complex(r, x))
    return zt + zmv, zt


def sequence_admittance(z1: complex, z0: complex) -> NDArray[np.complex128]:
    """Admittance (S) of a three-phase source of sequence impedances ``z1 = z2`` and ``z0``."""
    zs, zm = (2 * z1 + z0) / 3, (z0 - z1) / 3
    z = np.full((3, 3), zm, dtype=np.complex128) + np.eye(3) * (zs - zm)
    return np.asarray(np.linalg.inv(z), dtype=np.complex128)


def source_edit(vsource: str, z1: complex, z0: complex) -> list[str]:
    """The edit giving ``vsource`` the sequence impedances (ohm)."""
    z = f"[{z1.real!r}, {z1.imag!r}]"
    return [f"Edit {vsource} Z1={z} Z2={z} Z0=[{z0.real!r}, {z0.imag!r}]"]


def load_commands(site: Upstream) -> list[str]:
    """The lumped loads of the other feeders, at zero power (set per scenario)."""
    return [
        f"New {name} phases=1 bus1={site.bus1(k)} kV={site.kv!r} kW=0 kvar=0 model=1"
        for k, name in enumerate(site.loads)
    ]


def upstream_issues(networks: Sequence[Network]) -> list[str]:
    """The networks ``variants.upstream`` cannot serve (outside its scope), sorted."""
    return sorted(n.id for n in networks if not in_scope(n.id))


def upstream_of(
    network: Network, prepared: PreparedNetwork, catalog: Catalog, root: Path | None = None
) -> Upstream:
    """The substation of ``network`` (``prepared``: its unperturbed model); each sibling feeder
    compiled once in its own context. ``ValueError`` outside the scope, for a source that is not
    three-phase, or for a load that is not single-phase wye."""
    found = _ID.fullmatch(network.id)
    if found is None:
        raise ValueError(f"{network.id}: variants.upstream applies to csiro_enwl feeders only")
    siblings = tuple(
        sorted(
            other
            for other in catalog.networks
            if (m := _ID.fullmatch(other)) and m["net"] == found["net"] and other != network.id
        )
    )
    if not siblings:
        raise ValueError(f"{network.id}: no other feeder of its substation in the catalog")
    model = prepared.model
    vs = vsource_of(model)
    phase_nodes = tuple(model.nodes[k].opendss_node for k in vs.terminals[0] if k != GROUND)
    if len(phase_nodes) != 3:
        raise ValueError(f"{vs.name}: a three-phase source is needed, not {len(phase_nodes)}")
    bus = model.nodes[vs.terminals[0][0]].bus
    neutral = [n.opendss_node for n in model.nodes if n.bus == bus and n.conductor == "n"]
    count, _, _, kvs = _loads(prepared.ctx, network.id)
    if len(kvs) != 1:
        raise ValueError(f"{network.id}: loads of several rated voltages {sorted(kvs)}")
    p, q = np.zeros(3), np.zeros(3)
    for sid in siblings:
        ctx = DssContext.new()
        compile_network(ctx, catalog.network(sid), root)
        n, ps, qs, _ = _loads(ctx, sid)
        count, p, q = count + n, p + ps, q + qs
    with prepared.ctx.engine_call(f"Vsources {vs.name}"):
        prepared.ctx.api.Vsources.Name(vs.name.split(".", 1)[1])
        kv_ll = float(prepared.ctx.api.Vsources.BasekV())
    order = [int(node.rsplit(".", 1)[1]) - 1 for node in phase_nodes]  # source conductor -> phase
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"{vs.name}: on nodes {phase_nodes}, not phases 1, 2, 3")
    return Upstream(
        network.id, siblings, count, vs.name, phase_nodes,
        neutral[0].rsplit(".", 1)[1] if neutral else None, kvs.pop(), kv_ll,
        tuple(float(p[j]) for j in order), tuple(float(q[j]) for j in order),
    )  # fmt: skip


def _loads(ctx: DssContext, network_id: str) -> tuple[int, np.ndarray, np.ndarray, set[float]]:
    """Enabled loads of a compiled network: count, nominal P (W) and Q (var) per phase 1-3 and
    the set of rated voltages (kV)."""
    api = ctx.api
    p, q, kvs, n = np.zeros(3), np.zeros(3), set(), 0
    with ctx.engine_call(f"Loads of {network_id}"):
        i = api.Loads.First()
        while i:
            if api.CktElement.Enabled():
                phases = [k for k in api.CktElement.NodeOrder() if k in (1, 2, 3)]
                if len(phases) != 1 or api.Loads.IsDelta():
                    raise ValueError(f"{network_id}: Load.{api.Loads.Name()} is not single-phase "
                                     "wye")  # fmt: skip
                p[phases[0] - 1] += float(api.Loads.kW()) * 1e3
                q[phases[0] - 1] += float(api.Loads.kvar()) * 1e3
                kvs.add(float(api.Loads.kV()))
                n += 1
            i = api.Loads.Next()
    return n, p, q, kvs
