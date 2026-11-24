"""Tests for network recipes: master cut, extra commands, base loads at the coincident peak."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.networks.catalog import CatalogError, Network, load_catalog
from gridfm_datakit_distr.networks.compile import PREFIX_SUFFIX, compile_network, master_prefix
from gridfm_datakit_distr.normalize.loads import LoadNormalisationError, loads_at_coincident_peak
from gridfm_datakit_distr.scenarios.profiles.network_loadshapes import read_series

# A master in the EDP style: network, then monitors and a daily simulation that writes files.
NETWORK = """\
Clear\r
New Circuit.recipe basekv=0.4 pu=1.0 phases=3 bus1=src\r
New Line.l1 bus1=src bus2=b phases=3 r1=0.01 x1=0.01 r0=0.03 x0=0.03 c1=0 c0=0 length=1\r
New Loadshape.mult npts=4 interval=1 mult=(0.5 1.0 1.5 1.0) qmult=(0.1 0.2 0.3 0.2)\r
New Loadshape.actual npts=2 interval=1 Pmult=(3 5) useactual=yes\r
New Loadshape.ponly npts=2 interval=1 mult=(1 3)\r
New Load.a bus1=b.1 phases=1 kV=0.23 kW=2 kvar=1 model=1 daily=mult\r
New Load.b bus1=b.2 phases=1 kV=0.23 kW=1 kvar=0.5 model=1 daily=actual\r
New Load.c bus1=b.3 phases=1 kV=0.23 kW=1 kvar=0.5 model=1 daily=ponly\r
"""
SIMULATION = """\
New Monitor.m1 Load.a 1 mode=1\r
Set mode=daily number=24\r
Solve\r
Export Monitors m1\r
"""


def write(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    master = folder / "master.dss"
    master.write_bytes(text.encode())
    return master


def test_master_prefix_cuts_before_the_marker(tmp_path: Path) -> None:
    master = write(tmp_path, NETWORK + "  REDIRECT MonBus.dss \r\n" + SIMULATION)
    path = master_prefix(master, "redirect monbus.dss")
    assert path == master.with_name("master.dss" + PREFIX_SUFFIX)
    assert path.read_bytes() == NETWORK.encode()  # bytes as in the source, CRLF kept
    before = path.stat().st_mtime_ns
    assert master_prefix(master, "redirect MonBus.dss") == path
    assert path.stat().st_mtime_ns == before  # unchanged content is not rewritten
    path.write_bytes(b"stale")
    master_prefix(master, "redirect MonBus.dss")
    assert path.read_bytes() == NETWORK.encode()


@pytest.mark.parametrize("marker_lines", [0, 2])
def test_master_prefix_needs_exactly_one_marker(tmp_path: Path, marker_lines: int) -> None:
    master = write(tmp_path, NETWORK + "redirect MonBus.dss\r\n" * marker_lines)
    with pytest.raises(CatalogError, match=f"occurs {marker_lines} times"):
        master_prefix(master, "redirect MonBus.dss")


def test_compile_network_follows_the_recipe(tmp_path: Path) -> None:
    write(tmp_path / "src", NETWORK + "redirect MonBus.dss\r\n" + SIMULATION)
    network = Network(
        id="recipe", source="src", master="master.dss",
        master_stop_before="redirect MonBus.dss", base_loads="loadshape_coincident_peak",
        extra_commands=["Set VoltageBases=[0.4]"],
    )  # fmt: skip
    ctx = DssContext.new()
    record = compile_network(ctx, network, root=tmp_path)
    assert record.master.name == "master.dss" + PREFIX_SUFFIX
    assert record.extra_commands == ("Set VoltageBases=[0.4]",)
    assert record.load_actions == ("CalcVoltageBases",)
    assert int(ctx.api.Monitors.Count()) == 0  # nothing after the cut ran
    assert int(ctx.api.Solution.Mode()) == 0  # still Snapshot
    assert record.base_loads is not None
    assert record.base_loads.n_loads == 3
    assert not list(tmp_path.glob("src/*.csv"))  # no export


def test_base_loads_are_the_coincident_peak(tmp_path: Path) -> None:
    ctx = DssContext.new()
    ctx.load(write(tmp_path, NETWORK))
    record = loads_at_coincident_peak(ctx)
    api = ctx.api
    # The means of what OpenDSS simulates along each curve: a 2.0 kW, b 4.0, c 2.0 (sum 8.0). The
    # window is the shortest shape (2 points): summed P (1 + 3 + 1, 2 + 5 + 3) = (5, 10) kW, so c is
    # 10 / 8 for P and Q alike.
    c = 10.0 / 8.0
    api.Loads.Name("a")  # multipliers: mean P 1.0 x 2 kW, mean Q 0.2 x 1 kvar
    assert (api.Loads.kW(), api.Loads.kvar()) == pytest.approx((2.0 * c, 0.2 * c))
    api.Loads.Name("b")  # actual P (3, 5) without Q: OpenDSS set kW 5, kvar 0 (PF 1) on load
    assert (api.Loads.kW(), api.Loads.kvar()) == pytest.approx((4.0 * c, 0.0))
    api.Loads.Name("c")  # multipliers without Q: kvar follows P at kvar/kW = 0.5
    assert (api.Loads.kW(), api.Loads.kvar()) == pytest.approx((2.0 * c, 1.0 * c))
    assert record.total_kw_before == pytest.approx(2.0 + 5.0 + 1.0)
    assert (record.total_kw, record.total_kvar) == pytest.approx((10.0, 1.2 * c))
    assert record.coincident_peak_ratio == pytest.approx(c)
    assert record.window_h == pytest.approx(2.0)  # interval=1 is 1 h


def test_one_point_shapes_are_constants_in_the_window(tmp_path: Path) -> None:
    one_point = "New Loadshape.flat npts=1 interval=1 mult=(4)\r\n"
    load_d = "New Load.d bus1=b.1 phases=1 kV=0.23 kW=1 kvar=0 model=1 daily=flat\r\n"
    ctx = DssContext.new()
    ctx.load(write(tmp_path, NETWORK + one_point + load_d))
    record = loads_at_coincident_peak(ctx)
    # d adds 4 kW to every point of the window: (9, 14) kW against means 8 + 4
    assert record.coincident_peak_ratio == pytest.approx(14.0 / 12.0)
    ctx.api.Loads.Name("d")
    assert ctx.api.Loads.kW() == pytest.approx(4.0 * 14.0 / 12.0)


def test_profiles_keep_the_measured_values(tmp_path: Path) -> None:
    """The base loads change; the series the profile providers draw from do not."""
    write(tmp_path / "src", NETWORK)
    network = Network(
        id="recipe", source="src", master="master.dss", base_loads="loadshape_coincident_peak"
    )
    series = read_series(network, root=tmp_path)
    assert series.names == ("Load.a", "Load.b", "Load.c")
    # the shapes' first two points in W: 2 kW x (0.5, 1), actual (3, 5) kW, 1 kW x (1, 3)
    np.testing.assert_array_equal(series.p_w, [[1e3, 2e3], [3e3, 5e3], [1e3, 3e3]])


def test_a_load_without_loadshape_is_an_error(tmp_path: Path) -> None:
    ctx = DssContext.new()
    ctx.load(write(tmp_path, NETWORK + "New Load.d bus1=b.3 phases=1 kV=0.23 kW=1\r\n"))
    with pytest.raises(LoadNormalisationError, match=r"Load\.d has no daily or yearly loadshape"):
        loads_at_coincident_peak(ctx)


def test_edp_networks_have_a_recipe() -> None:
    catalog = load_catalog()
    for member in ("industrial", "rural", "urban"):
        network = catalog.network(f"edp/{member}")
        assert network.master_stop_before is not None
        assert network.base_loads == "loadshape_coincident_peak"
        assert "Set VoltageBases=[22 0.4]" in network.extra_commands
    assert catalog.network("edp/industrial").neutral == "mixed"
