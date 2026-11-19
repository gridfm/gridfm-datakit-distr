"""Power balance of a positive-sequence export, as gridfm-datakit computes it."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gridfm_datakit_distr.views.posseq import BASE_MVA


def power_balance(bus: pa.Table, branch: pa.Table, sn_mva: float = BASE_MVA) -> NDArray[np.float64]:
    """Largest |P| or |Q| mismatch (MW, MVAr) of every scenario: datakit's
    ``utils/power_balance.compute_bus_balance`` (injection - branch flows - shunt), per scenario
    in the order of first appearance in ``bus``."""
    sid = bus.column("load_scenario_idx").to_numpy()
    scenarios, first = np.unique(sid, return_index=True)
    scenarios = scenarios[np.argsort(first)]
    out = np.zeros(len(scenarios))
    bsid = branch.column("load_scenario_idx").to_numpy()
    for k, s in enumerate(scenarios):
        rows = sid == s
        n = int(rows.sum())
        col = {
            c: bus.column(c).to_numpy()[rows] for c in ("Pd", "Qd", "Pg", "Qg", "Vm", "GS", "BS")
        }
        brows = bsid == s
        f = branch.column("from_bus").to_numpy()[brows]
        t = branch.column("to_bus").to_numpy()[brows]
        p_out = np.bincount(f, branch.column("pf").to_numpy()[brows], n) + np.bincount(
            t, branch.column("pt").to_numpy()[brows], n
        )
        q_out = np.bincount(f, branch.column("qf").to_numpy()[brows], n) + np.bincount(
            t, branch.column("qt").to_numpy()[brows], n
        )
        v2 = col["Vm"] ** 2
        p_mis = col["Pg"] - col["Pd"] - p_out - col["GS"] * v2 * sn_mva
        q_mis = col["Qg"] - col["Qd"] - q_out + col["BS"] * v2 * sn_mva
        out[k] = max(np.abs(p_mis).max(), np.abs(q_mis).max())
    return out
