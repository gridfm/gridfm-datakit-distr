"""Tests for the load normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.normalize.loads import normalize_loads
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from tests.circuits import HEAVY, write_circuit

HEAVY_LOADS = {"Load.a3", "Load.b1", "Load.bd", "Load.bz", "Load.bi"}


def compiled(tmp_path: Path, files: dict[str, str]) -> DssContext:
    ctx = DssContext.new()
    ctx.load(write_circuit(tmp_path, files))
    ctx.set_options(tolerance=1e-10, max_iterations=200)
    return ctx


def load_models(ctx: DssContext) -> dict[str, int]:
    api = ctx.api
    models = {}
    for name in ("a3", "b1", "bd", "bz", "bi", "off"):
        api.Loads.Name(name)
        models[name] = int(api.Loads.Model())
    return models


def test_loads_draw_their_setpoints_after_normalisation_and_not_before(tmp_path: Path) -> None:
    ctx = compiled(tmp_path, HEAVY)
    assert ctx.solve().converged
    # Phase 1 sits at 0.93 pu at bus a and 0.89 pu at bus b: model-1 loads are below the default
    # vminpu = 0.95; the model-2 and model-5 loads follow their own models.
    before = setpoint_report(ctx)
    assert set(before.off) == HEAVY_LOADS
    assert before.max_rel > 0.01
    normalize_loads(ctx)
    assert ctx.solve().converged
    after = setpoint_report(ctx)
    assert after.passes()
    assert after.n_checked == len(HEAVY_LOADS)
    assert after.max_rel < 1e-9  # Tolerance 1e-10 (L1: 0.1-1.7 x Tolerance)


def test_normalisation_record(tmp_path: Path) -> None:
    ctx = compiled(tmp_path, HEAVY)
    record = normalize_loads(ctx)
    assert record.n_loads == len(HEAVY_LOADS)  # the disabled load is left alone
    assert record.models_before == {1: 3, 2: 1, 5: 1}
    assert record.commands[0] == "Edit Load.a3 model=1 vminpu=0.6 vmaxpu=1.5"
    assert load_models(ctx) == {"a3": 1, "b1": 1, "bd": 1, "bz": 1, "bi": 1, "off": 2}


def test_as_is_keeps_the_models(tmp_path: Path) -> None:
    ctx = compiled(tmp_path, HEAVY)
    record = normalize_loads(ctx, model="as_is", vminpu=0.7, vmaxpu=1.3)
    assert "Edit Load.bz vminpu=0.7 vmaxpu=1.3" in record.commands
    assert load_models(ctx) == {"a3": 1, "b1": 1, "bd": 1, "bz": 2, "bi": 5, "off": 2}
    ctx.api.Loads.Name("bi")
    assert (ctx.api.Loads.Vminpu(), ctx.api.Loads.Vmaxpu()) == (0.7, 1.3)


@pytest.mark.parametrize(("vminpu", "vmaxpu"), [(1.0, 1.5), (0.6, 0.99), (0.0, 1.5)])
def test_window_must_contain_one(vminpu: float, vmaxpu: float) -> None:
    with pytest.raises(ValueError, match="vminpu < 1 < vmaxpu"):
        normalize_loads(DssContext.new(), vminpu=vminpu, vmaxpu=vmaxpu)
