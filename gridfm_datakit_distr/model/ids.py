"""Identifiers of networks, variants, scenarios and nodes (outputs.md §1)."""

from __future__ import annotations

BASE_TAG = "base"
"""Variant tag of an unperturbed network as loaded from its source (conformance, inspection)."""

NETWORK_BITS, VARIANT_BITS, SCENARIO_BITS = 16, 16, 31
"""Bit fields of ``scenario_id = (i << 47) | (k << 31) | s``."""
MAX_NETWORK_INDEX = (1 << NETWORK_BITS) - 1
MAX_VARIANT_INDEX = 9999
"""Variant ids keep four digits (``v{k:04d}``), inside the 16-bit field."""
MAX_SCENARIO_INDEX = (1 << SCENARIO_BITS) - 1


def variant_id(network_id: str, k: int) -> str:
    """``"{network_id}/v{k:04d}"`` — the k-th sampled variant of a network."""
    if not 0 <= k <= MAX_VARIANT_INDEX:
        raise ValueError(f"variant index must be in [0, {MAX_VARIANT_INDEX}], got {k}")
    return f"{network_id}/v{k:04d}"


def scenario_id(i: int, k: int, s: int) -> int:
    """Dataset-unique int64 of scenario ``s`` of variant ``k`` of the ``i``-th network (networks
    ordered by id): ``(i << 47) | (k << 31) | s``."""
    for name, value, top in (
        ("network index", i, MAX_NETWORK_INDEX),
        ("variant index", k, MAX_VARIANT_INDEX),
        ("scenario index", s, MAX_SCENARIO_INDEX),
    ):
        if not 0 <= value <= top:
            raise ValueError(f"{name} must be in [0, {top}], got {value}")
    return (i << (VARIANT_BITS + SCENARIO_BITS)) | (k << SCENARIO_BITS) | s


def split_scenario_id(sid: int) -> tuple[int, int, int]:
    """The (network, variant, scenario) indices of a scenario id."""
    if not 0 <= sid < 1 << (NETWORK_BITS + VARIANT_BITS + SCENARIO_BITS):
        raise ValueError(f"not a scenario id: {sid}")
    i = sid >> (VARIANT_BITS + SCENARIO_BITS)
    k = (sid >> SCENARIO_BITS) & ((1 << VARIANT_BITS) - 1)
    s = sid & MAX_SCENARIO_INDEX
    return i, k, s


def base_variant_id(network_id: str) -> str:
    """``"{network_id}/base"`` — the unperturbed network."""
    return f"{network_id}/{BASE_TAG}"


def split_variant_id(variant: str) -> tuple[str, str]:
    """Split a variant id into (network_id, tag); network ids may contain '/'."""
    network_id, sep, tag = variant.rpartition("/")
    if not sep or not network_id or not (tag == BASE_TAG or (tag[:1] == "v" and tag[1:].isdigit())):
        raise ValueError(f"not a variant id: {variant!r}")
    return network_id, tag
