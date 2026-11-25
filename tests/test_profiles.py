"""Tests for the load profile providers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from gridfm_datakit_distr.config.schema import Profiles, SyntheticProfiles
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import prepare_network
from gridfm_datakit_distr.normalize.loads import normalize_loads
from gridfm_datakit_distr.scenarios.profiles import (
    Customer,
    ProfileError,
    customers_of,
    make_provider,
    profile_issues,
)
from gridfm_datakit_distr.scenarios.profiles.network_loadshapes import (
    LoadshapeProvider,
    read_series,
)
from gridfm_datakit_distr.scenarios.profiles.synthetic import SyntheticProvider
from gridfm_datakit_distr.scenarios.seeds import ScenarioStream, scenario_rng
from tests.circuits import LATERAL, PROFILED, write_circuit

SEED = 3
TAN_09 = math.tan(math.acos(0.9))


@pytest.fixture(scope="module")
def profiled(tmp_path_factory: pytest.TempPathFactory) -> tuple[Network, Path]:
    root = tmp_path_factory.mktemp("profiled")
    write_circuit(root / "src", PROFILED)
    return Network(id="t/prof", source="src", master="master.dss"), root


# ------------------------------------------------------------------- loadshape series


def test_series_follow_the_engine_semantics(profiled: tuple[Network, Path]) -> None:
    network, root = profiled
    series = read_series(network, root)
    assert series.interval_s == 3600
    assert series.times_s.tolist() == [3600 * k for k in range(1, 7)]  # 6-hour window
    at3 = series.at(3 * 3600)  # the third point of each shape (point k at k intervals)
    assert at3["Load.a"] == pytest.approx((3000.0, 300.0))  # actual kW and kvar
    assert at3["Load.b"] == pytest.approx((600.0, 2000 * TAN_09 * 0.3))  # kW x mult, PF kept
    assert set(at3) == {"Load.a", "Load.b"}
    cov = series.coverage(["Load.a", "Load.b", "Load.c", "Load.d"])
    assert (cov.n_customers, cov.n_mapped) == (4, 2)
    assert cov.constant == ("Load.c",)  # a one-point shape is not a profile
    assert cov.unmapped == ("Load.d",)
    assert cov.share == 0.5


def test_a_network_without_loadshapes_is_refused(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src", LATERAL)
    with pytest.raises(ProfileError, match="no load has a daily or yearly loadshape"):
        read_series(Network(id="t/lat", source="src", master="master.dss"), tmp_path)


def test_the_provider_check_names_the_networks_it_cannot_serve(tmp_path: Path) -> None:
    write_circuit(tmp_path / "src" / "prof", PROFILED)
    write_circuit(tmp_path / "src" / "lat", LATERAL)
    prof = Network(id="t/prof", source="src", master="prof/master.dss")
    lat = Network(id="t/lat", source="src", master="lat/master.dss")
    shapes = Profiles.model_validate({"provider": "network_loadshapes"})
    assert profile_issues(shapes, [prof, lat], tmp_path) == [
        "t/lat: no load has a daily or yearly loadshape"
    ]
    assert profile_issues(Profiles(), [prof, lat], tmp_path) == []  # synthetic: no loadshape


def test_generate_refuses_a_provider_it_cannot_serve_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With and without ``--dry-run``: exit 2, as an invalid config, naming the networks."""
    from typer.testing import CliRunner

    from gridfm_datakit_distr.cli import app
    from gridfm_datakit_distr.networks import cache, catalog
    from tests.test_executor import catalog as test_catalog

    monkeypatch.setattr(catalog, "load_catalog", lambda: test_catalog(tmp_path))
    monkeypatch.setattr(cache, "is_fetched", lambda source, root=None: True)
    config = tmp_path / "c.yaml"
    config.write_text(
        "dataset_name: d\nnetworks: {include: ['t/*']}\nsettings: {seed: 1}\nscenarios:\n"
        "  n_scenarios_per_variant: 1\n  profiles: {provider: edp_smart_meter}\n"
    )
    data = tmp_path / "data"
    args = ["generate", str(config), "--cache", str(tmp_path), "--data-dir", str(data)]
    for dry_run in ([], ["--dry-run"]):
        result = CliRunner().invoke(app, [*args, *dry_run])
        assert result.exit_code == 2, result.output
        for network in ("t/feeder", "t/reg"):
            expected = f"- scenarios.profiles.provider: {network}: no load has a daily or yearly"
            assert expected in result.output
        assert "dataset d" not in result.output  # the dry run's report
    assert not data.exists()


def provider(profiled: tuple[Network, Path], same_timestamp: bool = True) -> LoadshapeProvider:
    network, root = profiled
    customers = customers_of(prepare_network(network, root=root))
    config = Profiles.model_validate(
        {"provider": "network_loadshapes", "same_timestamp": same_timestamp}
    )
    p = make_provider(config, customers, network, root)
    assert isinstance(p, LoadshapeProvider)
    return p


def test_loadshape_draws(profiled: tuple[Network, Path]) -> None:
    p = provider(profiled)
    names = [c.element for c in p.customers]
    assert names == ["Load.a", "Load.b", "Load.c", "Load.d"]
    draw = p.draw(SEED, 0, 0, 0)
    assert draw == p.draw(SEED, 0, 0, 0)
    assert draw.time_s in {3600 * k for k in range(1, 7)}
    assert draw.customer_times_s is None
    at = p.series.at(draw.time_s)
    assert draw.p_w[:2].tolist() == pytest.approx([at["Load.a"][0], at["Load.b"][0]])
    # c keeps its one point, d its nominal values
    assert draw.p_w[2:].tolist() == pytest.approx([2100.0, 1500.0])
    assert draw.q_var[2:].tolist() == pytest.approx([700.0, 500.0])
    times = {p.draw(SEED, 0, 0, s).time_s for s in range(60)}
    assert times == {3600 * k for k in range(1, 7)}


def test_independent_times_per_customer(profiled: tuple[Network, Path]) -> None:
    p = provider(profiled, same_timestamp=False)
    draws = [p.draw(SEED, 0, 0, s) for s in range(40)]
    assert all(d.time_s is None and d.customer_times_s is not None for d in draws)
    assert any(len(set(d.customer_times_s.tolist()[:2])) == 2 for d in draws)  # type: ignore[union-attr]
    # PV draws its hour from PROFILE_TIME when there is no common time: not the customers' stream
    pv_hour = [
        scenario_rng(SEED, 0, 0, s, ScenarioStream.PROFILE_TIME).uniform(0, 24) for s in range(40)
    ]
    for c in range(len(p.customers)):
        times = [d.customer_times_s[c] for d in draws]  # type: ignore[index]
        assert abs(np.corrcoef(pv_hour, times)[0, 1]) < 0.5, c


def test_the_engine_draws_the_same_values(profiled: tuple[Network, Path]) -> None:
    """OpenDSS's own time simulation at hour 3 delivers what the provider gives for 3 h."""
    network, root = profiled
    p = provider(profiled)
    prepared = prepare_network(network, root=root)
    ctx, api = prepared.ctx, prepared.ctx.api
    normalize_loads(ctx)
    ctx.command("Set mode=yearly stepsize=1h number=1")
    ctx.command("Set time=(2,0)")  # a step advances the clock before solving
    ctx.command("Solve")
    assert api.Solution.DblHour() == pytest.approx(3.0)
    expected = p.series.at(3 * 3600)
    for name, (p_w, q_var) in expected.items():
        api.Circuit.SetActiveElement(name)
        s = np.asarray(api.CktElement.Powers())
        assert (s[0::2].sum() * 1e3, s[1::2].sum() * 1e3) == pytest.approx((p_w, q_var), rel=1e-8)


# ------------------------------------------------------------------- synthetic


CUSTOMERS = (Customer("Load.x", 1000.0, 200.0), Customer("Load.y", 3000.0, 0.0))


def synthetic(**changes: object) -> SyntheticProvider:
    spec = SyntheticProfiles.model_validate(changes)
    return SyntheticProvider(CUSTOMERS, spec)


def test_synthetic_statistics_match_the_spec() -> None:
    p = synthetic()  # global U(0.1, 1.2), sigma 0.3, PF U(0.9, 0.99)
    draws = [p.draw(SEED, 0, 0, s) for s in range(3000)]
    nominal = np.array([c.p_w for c in CUSTOMERS])
    ratio = np.array([d.p_w / nominal for d in draws])  # g x m per customer
    g = np.array([d.global_scale for d in draws])
    assert stats.kstest(g, stats.uniform(0.1, 1.1).cdf).pvalue > 1e-3
    noise = np.log(ratio / g[:, None])  # log m ~ N(-sigma^2/2, sigma^2): mean-one lognormal
    for column in noise.T:
        assert stats.kstest(column, stats.norm(-0.045, 0.3).cdf).pvalue > 1e-3
    pf = np.array([d.p_w / np.hypot(d.p_w, d.q_var) for d in draws])
    assert pf.min() >= 0.9
    assert pf.max() < 0.99
    assert all(d.time_s is None for d in draws)


def test_synthetic_draws_are_positional_and_independent() -> None:
    a = synthetic().draw(SEED, 1, 2, 3)
    assert a == synthetic().draw(SEED, 1, 2, 3)
    assert a != synthetic().draw(SEED, 1, 2, 4)
    b = synthetic(local_sigma=0.0).draw(SEED, 1, 2, 3)
    assert b.global_scale == a.global_scale  # the noise has its own stream
    assert b.p_w.tolist() == pytest.approx([a.global_scale * c.p_w for c in CUSTOMERS])


def test_the_load_level_scales_every_nominal_load() -> None:
    a = synthetic().draw(SEED, 1, 2, 3)
    b = SyntheticProvider(CUSTOMERS, SyntheticProfiles(), 1.7).draw(SEED, 1, 2, 3)
    assert b.global_scale == a.global_scale  # g stays the scenario's factor
    np.testing.assert_allclose(b.p_w, 1.7 * a.p_w, rtol=1e-15)
    np.testing.assert_allclose(b.q_var, 1.7 * a.q_var, rtol=1e-15)
    assert SyntheticProvider(CUSTOMERS, SyntheticProfiles(), 1.0).draw(SEED, 1, 2, 3) == a


def test_customers_take_the_variants_nominal_values(profiled: tuple[Network, Path]) -> None:
    network, root = profiled
    customers = customers_of(prepare_network(network, root=root))
    by = {c.element: c for c in customers}
    assert by["Load.d"] == Customer("Load.d", 1500.0, 500.0)
    assert by["Load.b"].q_var == pytest.approx(2000 * TAN_09)
