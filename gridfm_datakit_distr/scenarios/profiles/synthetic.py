"""Synthetic profiles.

Per scenario a global factor ``g`` (``global_scale`` spec) and per customer a mean-one lognormal
factor ``m = exp(sigma·z - sigma²/2)``, ``z ~ N(0, 1)``, so that ``E[P] = E[g] · P_nominal``; per
customer a power factor ``pf`` (spec): ``P = g · m · P_nominal``, ``Q = P · tan(acos(pf))``
(lagging, the sign of ``P``). Each quantity has its own stream.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gridfm_datakit_distr.config.schema import SyntheticProfiles
from gridfm_datakit_distr.scenarios.profiles.base import Customer, LoadDraw
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, customers: Sequence[Customer], spec: SyntheticProfiles) -> None:
        self.customers = tuple(customers)
        self.spec = spec
        self._p_nominal = np.array([c.p_w for c in self.customers], dtype=np.float64)

    def draw(self, seed: int, net: int, k: int, s: int) -> LoadDraw:
        n = len(self.customers)
        g = float(
            self.spec.global_scale.sample(scenario_rng(seed, net, k, s, ScenarioStream.LOAD_SCALE))
        )
        sigma = self.spec.local_sigma
        z = scenario_rng(seed, net, k, s, ScenarioStream.LOAD_NOISE).standard_normal(n)
        m = np.exp(sigma * z - sigma**2 / 2)
        rng = scenario_rng(seed, net, k, s, ScenarioStream.POWER_FACTOR)
        pf = np.array([float(self.spec.power_factor.sample(rng)) for _ in range(n)])
        p = g * m * self._p_nominal
        q = p * np.tan(np.arccos(pf))
        return LoadDraw(p_w=p, q_var=q, global_scale=g)
