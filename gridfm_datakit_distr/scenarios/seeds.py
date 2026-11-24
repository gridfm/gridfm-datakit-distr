"""Seed tree.

Every random draw comes from a :class:`numpy.random.SeedSequence` fixed by its position in the
tree, never by the order in which work is done:

    variant (n, k)           SeedSequence(seed, spawn_key=(n0, n1, k))
    scenario (n, k, s)       SeedSequence(seed, spawn_key=(n0, n1, k, s))   variant.spawn[s]
    variant stream p, try a  SeedSequence(seed, spawn_key=(n0, n1, k, 2**31 + p, a))
    scenario stream p        SeedSequence(seed, spawn_key=(n0, n1, k, s, p))

``n`` is the network's key, :func:`network_key` of its id, split into two 32-bit words ``n0``,
``n1``: a network's draws do not depend on the other networks of the dataset. ``k`` is the variant
index, ``s`` the scenario index. Each quantity draws from its own stream (:class:`VariantStream`,
:class:`ScenarioStream`), so enabling or resizing one quantity never shifts the draws of another;
a variant's streams sit above the scenario indices (``s < 2**31``) and never share a key with a
scenario. The ``seed`` columns (outputs.md §4, §5) are 64-bit fingerprints of a node's sequence; the
draws come from the sequence itself (128-bit pool).
"""

from __future__ import annotations

import hashlib
from enum import IntEnum

import numpy as np

from gridfm_datakit_distr.model.ids import MAX_SCENARIO_INDEX, MAX_VARIANT_INDEX, SCENARIO_BITS

_VARIANT_STREAM_BASE = 1 << SCENARIO_BITS


class VariantStream(IntEnum):
    """Random quantities drawn once per variant. Codes are provenance: append, never renumber."""

    LINE_LENGTH = 0
    NEUTRAL_IMPEDANCE = 1
    GROUNDING_RESISTANCE = 2
    PHASE_REALLOCATION = 3
    RECONFIGURATION = 4
    DER_PLACEMENT = 5
    DER_SIZE = 6
    DER_POWER_FACTOR = 7
    DER_REACTIVE_FRACTION = 8
    DER_REACTIVE_THRESHOLD = 9
    STORAGE_PLACEMENT = 10
    """Which PV customers get a storage unit (``der.storage.share``)."""
    STORAGE_SIZE = 11
    """Power rating of each storage unit, drawn only when ``der.storage.kw`` is a value spec."""


class ScenarioStream(IntEnum):
    """Random quantities drawn once per scenario. Codes are provenance: append, never renumber."""

    PROFILE_TIME = 0
    LOAD_SCALE = 1
    LOAD_NOISE = 2
    POWER_FACTOR = 3
    PV_OUTPUT = 4
    SOURCE_VOLTAGE = 5
    TAP_JITTER = 6
    CUSTOMER_TIMES = 7
    """Per-customer profile times (``same_timestamp: false``); PV keeps ``PROFILE_TIME``."""
    OPF_COSTS = 8
    """Cost coefficients of the OPF's preset ``sampled``."""
    PV_AVAILABILITY = 9
    """Availability of each PV unit (``scenarios.pv_profile.availability``)."""
    STORAGE_SETPOINT = 10
    """Set-point of each storage unit (``scenarios.storage_profile.setpoint``)."""
    DG_SETPOINT = 11
    """Set-point of each dispatchable generator (``scenarios.dg_profile.setpoint``)."""


def network_key(network_id: str) -> int:
    """The 64-bit key of a network's random sequences: the first 8 bytes of the SHA-256 of its id
    (big-endian)."""
    return int.from_bytes(hashlib.sha256(network_id.encode("utf-8")).digest()[:8], "big")


def _words(net: int) -> tuple[int, int]:
    return net >> 32, net & 0xFFFF_FFFF


def _check(seed: int, net: int, k: int, s: int | None = None) -> None:
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    bounds = [("network key", net, 2**64 - 1), ("variant index", k, MAX_VARIANT_INDEX)]
    if s is not None:
        bounds.append(("scenario index", s, MAX_SCENARIO_INDEX))
    for name, value, top in bounds:
        if not 0 <= value <= top:
            raise ValueError(f"{name} must be in [0, {top}], got {value}")


def variant_sequence(seed: int, net: int, k: int) -> np.random.SeedSequence:
    """The sequence of variant ``k`` of the network of key ``net``."""
    _check(seed, net, k)
    return np.random.SeedSequence(seed, spawn_key=(*_words(net), k))


def scenario_sequence(seed: int, net: int, k: int, s: int) -> np.random.SeedSequence:
    """The sequence of scenario ``s`` of variant ``k`` of the network of key ``net``."""
    _check(seed, net, k, s)
    return np.random.SeedSequence(seed, spawn_key=(*_words(net), k, s))


def variant_rng(
    seed: int, net: int, k: int, stream: VariantStream, attempt: int = 0
) -> np.random.Generator:
    """Generator of one per-variant quantity; ``attempt`` numbers the resamplings of a variant
    that failed its checks."""
    _check(seed, net, k)
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")
    key = (*_words(net), k, _VARIANT_STREAM_BASE + int(stream), attempt)
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def scenario_rng(
    seed: int, net: int, k: int, s: int, stream: ScenarioStream
) -> np.random.Generator:
    """Generator of one per-scenario quantity."""
    _check(seed, net, k, s)
    key = (*_words(net), k, s, int(stream))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def variant_seed(seed: int, net: int, k: int) -> int:
    """The ``seed`` column of a variant: 64-bit fingerprint of its sequence."""
    return int(variant_sequence(seed, net, k).generate_state(1, np.uint64)[0])


def scenario_seed(seed: int, net: int, k: int, s: int) -> int:
    """The ``seed`` column of a scenario: 64-bit fingerprint of its sequence."""
    return int(scenario_sequence(seed, net, k, s).generate_state(1, np.uint64)[0])
