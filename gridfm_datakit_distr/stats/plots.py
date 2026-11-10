"""Plots of a dataset's statistics. Needs the ``plots`` extra (matplotlib).

One PNG per quantity in ``<dataset>/stats/``: the distribution of each network (up to
``MAX_CURVES``; beyond, all networks pooled) from the histograms of :mod:`stats.physics`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gridfm_datakit_distr.stats.physics import RANGES, NetworkStats

MAX_CURVES = 8
DISPLAY_BINS = 120
FIGURES = {
    "voltage": (("vm_a", "vm_b", "vm_c", "vm_s1", "vm_s2"), "|V| of phase nodes (pu)"),
    "neutral_voltage": (("neutral_v",), "|V| of neutral nodes to ground (pu)"),
    "vuf": (("vuf",), "voltage unbalance factor |V2|/|V1|"),
    "line_loading": (("line_loading",), "line loading (phase conductors, |I| / ampacity)"),
    "neutral_loading": (("neutral_loading",), "neutral loading (|I_n| / line ampacity)"),
    "transformer_loading": (("transformer_loading",), "transformer loading (|S| / rating)"),
}


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def write_plots(root: Path, stats: dict[str, NetworkStats]) -> list[Path]:
    """Write the figures; returns their paths (a figure without data is skipped)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "stats"
    out.mkdir(exist_ok=True)
    written = []
    pooled = len(stats) > MAX_CURVES
    for name, (quantities, label) in FIGURES.items():
        curves = []  # (label, low, high, counts) of every non-empty distribution
        for quantity in quantities:
            lo, hi, bins = RANGES["vm" if quantity.startswith("vm_") else quantity]
            if pooled:
                groups = {"all networks": list(stats.values())}
            else:
                groups = {nid: [s] for nid, s in sorted(stats.items())}
            for group, members in groups.items():
                counts = np.zeros(bins, dtype=np.int64)
                for s in members:
                    if quantity in s.histograms:
                        counts += s.histograms[quantity].counts[:bins]
                if counts.sum():
                    suffix = f" {quantity.removeprefix('vm_')}" if name == "voltage" else ""
                    curves.append((f"{group}{suffix}", lo, hi, counts))
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        # show the populated bins only, merged to at most DISPLAY_BINS
        filled = np.flatnonzero(np.sum([c for *_, c in curves], axis=0))
        first, last = int(filled[0]), int(filled[-1]) + 1
        step = max(1, -(-(last - first) // DISPLAY_BINS))
        for text, lo, hi, counts in curves:
            width = (hi - lo) / counts.size
            window = counts[first:last]
            merged = np.add.reduceat(window, np.arange(0, window.size, step))
            centres = lo + (first + np.arange(merged.size) * step + step / 2) * width
            ax.plot(centres, merged / counts.sum(), label=text, lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel(f"share of values per bin of {step * width:.3g}")
        ax.legend(fontsize="small")
        path = out / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
