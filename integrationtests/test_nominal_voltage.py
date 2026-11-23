"""Voltage limits per unit of the nominal voltage the catalog declares (CSIRO: 230 V)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gridfm_datakit_distr.config.load import load_config
from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network

pytestmark = [pytest.mark.network_data]

REPO = Path(__file__).parents[1]


def test_the_slice_config_gives_the_uk_limits_on_a_csiro_feeder(
    network_master: Callable[[str], Path],
) -> None:
    pytest.importorskip("casadi")
    from gridfm_datakit_distr.opf.support import vpn_bounds

    network_master("csiro_enwl/n05_f8")
    config = load_config(REPO / "scripts" / "config" / "opf_slice.yaml")
    limits = config.networks.voltage_limits
    model = prepare_network(load_catalog().network("csiro_enwl/n05_f8")).model
    assert {round(n.kv_base_ln, 5) for n in model.nodes} == {0.24018}  # the 0.416 kV base
    assert {n.kv_nominal_ln for n in model.nodes} == {0.23}
    _, lo, hi = vpn_bounds(model, limits.vmin_pu, limits.vmax_pu)
    np.testing.assert_allclose(lo, 216.2, rtol=1e-12)
    np.testing.assert_allclose(hi, 253.0, rtol=1e-12)


def test_networks_without_a_declared_nominal_keep_their_base(
    network_master: Callable[[str], Path],
) -> None:
    network_master("ieee13")
    model = prepare_network(load_catalog().network("ieee13")).model
    assert {n.kv_nominal_ln for n in model.nodes} == {None}
    assert all(n.nominal_v_ln == n.kv_base_ln * 1e3 for n in model.nodes)
