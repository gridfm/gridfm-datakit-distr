"""Network states: the taps of regulated transformers and the steps of controlled capacitors
(docs/manual/outputs.md §1 and §5).

``state_id`` is content-addressed: the first 16 hex digits of the SHA-1 of the variant id, the taps
and the capacitor step states. Taps are stored rounded to 1e-9 pu, so every route to a state (plain
settling, settling again from jittered taps, as compiled) applies the same taps and yields the same
Yprims bit for bit. The origin is the route of the scenario that produced the object, stored per
scenario (``scenario_meta.origin``): scenarios that share a state can reach it by different routes,
so it is not part of the state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

Origin = Literal["static_settle", "static_settle+jitter", "as_is"]
TAP_DECIMALS = 9


@dataclass(frozen=True, slots=True)
class NetworkState:
    variant_id: str
    taps: tuple[tuple[str, tuple[float, ...]], ...]
    """Regulated transformer -> tap of each winding (pu ratio), ordered by transformer."""
    capacitors: tuple[tuple[str, tuple[int, ...]], ...]
    """Controlled capacitor -> state of each step (0/1), ordered by capacitor."""
    origin: Origin

    def __post_init__(self) -> None:
        # routes to one tap position differ by an ulp (lo + n*step vs OpenDSS's own increments)
        taps = tuple((t, tuple(round(float(x), TAP_DECIMALS) for x in v)) for t, v in self.taps)
        object.__setattr__(self, "taps", taps)

    @property
    def state_id(self) -> str:
        content = json.dumps(
            [
                self.variant_id,
                [[t, list(taps)] for t, taps in self.taps],
                [[c, list(states)] for c, states in self.capacitors],
            ],
            separators=(",", ":"),
        )
        return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]

    def taps_json(self) -> str:
        """The ``taps`` column (outputs.md §5)."""
        return json.dumps({t: list(taps) for t, taps in self.taps}, sort_keys=True)

    def capacitors_json(self) -> str:
        """The ``capacitor_states`` column (outputs.md §5)."""
        return json.dumps({c: list(states) for c, states in self.capacitors}, sort_keys=True)
