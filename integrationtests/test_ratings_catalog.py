"""Line ampacities on catalog networks: kept where the source declares them, null elsewhere."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from gridfm_datakit_distr.networks.catalog import load_catalog
from gridfm_datakit_distr.networks.prepare import prepare_network

pytestmark = [pytest.mark.conformance, pytest.mark.network_data]


@pytest.mark.parametrize(
    ("network_id", "n_rated"),
    [
        ("ieee13", 0),
        ("csiro_enwl/n01_f1", 0),
        ("edp/rural", 0),
        # wire data and linecodes of IEEE 8500 declare 156 to 580 A on 1290 of its lines
        ("ieee8500", 1290),
    ],
)
def test_only_declared_ampacities_are_kept(
    network_master: Callable[[str], Path], network_id: str, n_rated: int
) -> None:
    network_master(network_id)
    model = prepare_network(load_catalog().network(network_id)).model
    lines = [e for e in model.elements if e.element_class in ("line", "switch")]
    rated = [e.rating_a for e in lines if e.rating_a is not None]
    assert len(rated) == n_rated
    assert all(r > 0 for r in rated)
