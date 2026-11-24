"""Profile providers on catalog networks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.config.schema import Profiles
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.scenarios.profiles import customers_of, make_provider
from gridfm_datakit_distr.scenarios.profiles.network_loadshapes import (
    LoadshapeProvider,
    read_series,
)

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]

ON_PEAK_S = 566 * 60
"""The on-peak snapshot of the IEEE European LV feeder: minute 566, 09:26 (pandapower's
``on_peak_566``; the IEEE documentation is not in the cache)."""


def test_european_lv_on_peak(network_master: Callable[[str], Path]) -> None:
    """At 566 min the provider gives what OpenDSS's own yearly simulation applies, and a snapshot
    with those setpoints solves to the simulation's voltages."""
    network_master("ieee_eulv")
    network = load_catalog().network("ieee_eulv")
    prepared = prepare_network(network)
    customers = customers_of(prepared)
    provider = make_provider(
        Profiles.model_validate({"provider": "network_loadshapes"}), customers, network
    )
    assert isinstance(provider, LoadshapeProvider)
    assert provider.coverage.n_mapped == len(customers) == 55
    assert provider.series.interval_s == 60
    assert provider.series.times_s[-1] == 1440 * 60
    on_peak = provider.series.at(ON_PEAK_S)

    # OpenDSS's yearly simulation, one-minute steps, with the normalised (constant power) loads
    ctx, api = prepared.ctx, prepared.ctx.api
    ctx.command("Set mode=yearly stepsize=1m number=1")
    ctx.command("Set time=(9,1500)")  # 09:25: the step solves at 09:26
    ctx.command("Solve")
    assert api.Solution.DblHour() * 60 == pytest.approx(566)
    for name, (p_w, q_var) in on_peak.items():
        api.Circuit.SetActiveElement(name)
        s = np.asarray(api.CktElement.Powers())
        assert (s[0::2].sum() * 1e3, s[1::2].sum() * 1e3) == pytest.approx((p_w, q_var), rel=1e-8)
    v_simulation = ctx.ynode_voltages()

    snapshot = prepare_network(network)
    for name, (p_w, q_var) in on_peak.items():
        snapshot.ctx.command(f"Edit {name} kW={p_w / 1e3!r} kvar={q_var / 1e3!r}")
    assert snapshot.ctx.solve().converged
    v = snapshot.ctx.ynode_voltages()
    base = np.array([n.kv_base_ln for n in snapshot.model.nodes]) * 1e3
    assert snapshot.ctx.ynode_order() == ctx.ynode_order()
    assert float(np.abs(v - v_simulation).max() / base.min()) < 1e-8


@pytest.mark.parametrize(
    ("network_id", "n_loads", "hours", "n_constant", "ratio"),
    [
        ("edp/rural", 2731, 168, 0, 1.4836),
        ("edp/industrial", 191, 480, 34, 1.7338),  # 34 meters with one reading; shapes of 480-744 h
        pytest.param("edp/urban", 35297, 24, 0, 1.3229, marks=pytest.mark.slow),
    ],
)
def test_edp_meter_data(
    network_master: Callable[[str], Path],
    network_id: str,
    n_loads: int,
    hours: int,
    n_constant: int,
    ratio: float,
) -> None:
    network_master(network_id)
    network = load_catalog().network(network_id)
    prepared = prepare_network(network)  # base loads: c times the mean of each load's shape
    customers = customers_of(prepared)
    provider = make_provider(
        Profiles.model_validate({"provider": "edp_smart_meter"}), customers, network
    )
    assert isinstance(provider, LoadshapeProvider)
    cov = provider.coverage
    assert cov.n_customers == n_loads
    assert len(cov.constant) == n_constant
    assert cov.unmapped == ()
    assert cov.n_mapped == n_loads - n_constant  # the unmapped ones are reported by name
    assert provider.series.interval_s == 3600
    assert provider.series.times_s[-1] == hours * 3600
    # The base loads add up to the measured coincident peak over the provider's window.
    series = provider.series
    measured = series.p_w.sum(axis=0) + sum(p for p, _ in series.constants.values())
    assert sum(c.p_w for c in customers) == pytest.approx(measured.max(), rel=1e-12)
    base = prepared.compiled.base_loads
    assert base is not None
    assert (base.coincident_peak_ratio, base.window_h) == pytest.approx((ratio, hours), rel=1e-4)
    if network_id != "edp/industrial":  # full series: c times their mean is the base load
        nominal = {c.element: (c.p_w, c.q_var) for c in customers}
        means = np.stack([series.p_w.mean(axis=1), series.q_var.mean(axis=1)], 1)
        expected = np.array([nominal[n] for n in series.names])
        c = base.coincident_peak_ratio
        np.testing.assert_allclose(c * means, expected, rtol=1e-12, atol=1e-9)


def test_networks_without_loadshapes_are_refused(network_master: Callable[[str], Path]) -> None:
    from gridfm_datakit_distr.scenarios.profiles import ProfileError

    network_master("csiro_enwl/n01_f1")
    with pytest.raises(ProfileError, match="no load has a daily or yearly loadshape"):
        read_series(load_catalog().network("csiro_enwl/n01_f1"))
