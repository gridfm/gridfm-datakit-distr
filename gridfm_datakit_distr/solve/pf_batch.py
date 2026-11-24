"""Per-scenario power-flow loop of one variant.

A :class:`VariantRunner` loads a variant once (catalog recipe plus its edit script, normalised) and
runs scenarios by index. Each scenario:

1. draws its loads (profile provider), PV outputs (at the profile's time of day), storage and
   DG set-points, source voltage and source unbalance from its own streams;
2. sets them (``Loads``/``Generators`` setters, ``Vsources.PU``, the negative-sequence current
   source of :func:`add_negative_sequence_source`);
3. resets the controls to the variant's compiled state, then settles or keeps them;
4. solves with ``ControlMode=OFF``; a non-converged solve is retried once with ``MaxIterations``
   x 5; then the setpoint check and, on a sample of scenarios, the KCL residual;
5. returns a :class:`ScenarioOutcome`: status, draws and, if converged at its setpoints, the node
   voltages (and the element currents with ``labels.element_currents``). Any exception becomes
   ``pf_status = error`` with its message: nothing is dropped.

Every solve starts with ``Set Mode=Snap``: OpenDSS then starts from the same point whatever was
solved before, so a scenario's result is bitwise independent of the order and of the process that
runs it.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.engine.context import DssContext, SolveStatus
from gridfm_datakit_distr.extract.injections import injections
from gridfm_datakit_distr.extract.source import (
    A,
    Source,
    negative_sequence,
    read_source,
    vsource_of,
)
from gridfm_datakit_distr.model.electrical import ElectricalModel
from gridfm_datakit_distr.model.ids import scenario_id
from gridfm_datakit_distr.model.state import NetworkState, Origin
from gridfm_datakit_distr.networks.catalog import Network
from gridfm_datakit_distr.networks.prepare import PreparedNetwork, PrepareOptions, prepare_network
from gridfm_datakit_distr.scenarios.controls import (
    MAX_CONTROL_ITER,
    StateRegistry,
    apply_state,
    has_controls,
    model_at_state,
    read_state,
    settle,
)
from gridfm_datakit_distr.scenarios.der import (
    DGSetpoints,
    PVDraw,
    PVOutputs,
    StorageSetpoints,
    dg_units,
    pv_units,
    storage_units,
)
from gridfm_datakit_distr.scenarios.profiles import customers_of, make_provider
from gridfm_datakit_distr.scenarios.pvgis import series_for
from gridfm_datakit_distr.scenarios.seeds import (
    ScenarioStream,
    network_key,
    scenario_rng,
    scenario_seed,
)
from gridfm_datakit_distr.solve.extraction import MAX_CACHED_STATES, ExtractionPlan, build_plan
from gridfm_datakit_distr.validate.residual import check_this_scenario, residual
from gridfm_datakit_distr.validate.setpoints import setpoint_report
from gridfm_datakit_distr.variants.generate import VariantResult

PFStatus = Literal["converged", "off_setpoint", "not_converged", "error"]
RETRY_FACTOR = 5
RAMP_STEPS = 10
"""Load-ramp fallback (``VariantRunner._ramp``): setpoints raised in this many equal steps."""
NEG_SEQUENCE_SOURCE = "source_neg_seq"
"""Name of the ``Isource`` that gives the voltage source its negative sequence."""


def prepare_options(config: GenerationConfig) -> PrepareOptions:
    """Normalisation and solver settings of a config."""
    loads, s_base = config.normalisation.loads, config.normalisation.s_base_va
    return PrepareOptions(
        load_model=loads.model,
        vminpu=loads.vminpu,
        vmaxpu=loads.vmaxpu,
        tolerance=config.settings.validation_tolerance,
        s_base_va=None if s_base == "auto" else s_base,
    )


def engine_indices(names: Iterable[str], elements: Iterable[str]) -> list[int]:
    """Engine indices (1-based, for ``Loads.Idx`` and ``Generators.Idx``) of ``<Class>.<name>``
    elements among the engine's ``AllNames()``."""
    index = {n.lower(): j for j, n in enumerate(names, start=1)}
    return [index[e.split(".", 1)[1].lower()] for e in elements]


def add_negative_sequence_source(ctx: DssContext, model: ElectricalModel) -> Source:
    """Add to the compiled circuit of ``model`` the current source that gives its voltage source
    a negative sequence, and return the voltage source (:func:`read_source`).

    A Thevenin source ``E_1 + E_2`` behind ``Y_s`` injects ``Y_s (E_1 + E_2 - V)``; the Vsource
    injects ``Y_s (E_1 - V)``, so an ideal current source at the same nodes injecting
    ``I_2 = Y_s E_2`` completes it, with no node and no admittance added (spike s58). For a
    negative-sequence ``E_2`` and a source built from sequence impedances ``I_2`` is a balanced
    negative-sequence set: one ``Isource ... sequence=neg``, zero until
    :func:`negative_sequence_current` sets it. Added after the extraction, which does not take
    it: the model keeps one source. ``ValueError`` unless the source has three conductors and an
    admittance that keeps the sequences apart."""
    source = read_source(ctx, model)
    i = source.y_s @ negative_sequence(source.e_v)
    neg = (i[0] + A**2 * i[1] + A * i[2]) / 3
    if max(abs(i.sum()) / 3, abs(i[0] + A * i[1] + A**2 * i[2]) / 3) > 1e-9 * abs(neg):
        raise ValueError(f"{model.variant_id}: the source admittance couples the sequences")
    vs = vsource_of(model)
    ctx.activate_element(vs.name)
    with ctx.engine_call(f"CktElement.BusNames {vs.name}"):
        bus1 = str(ctx.api.CktElement.BusNames()[0])
    name = f"Isource.{NEG_SEQUENCE_SOURCE}"
    ctx.command(f"New {name} bus1={bus1} phases=3 amps=0 angle=0 sequence=neg")
    return source


def negative_sequence_current(source: Source, v_src: float, u2: float, angle_rad: float) -> complex:
    """Current (A) of the first conductor of the negative-sequence current source for a source
    voltage ``v_src`` (pu) with ratio ``u2`` and angle ``angle_rad``: ``(Y_s E_2)_0``."""
    e2 = v_src * u2 * np.exp(1j * angle_rad) * negative_sequence(source.e_v)
    return complex((source.y_s @ e2)[0])


def set_setpoints(
    api: Any,
    load_idx: Sequence[int],
    load_p_w: NDArray[np.float64],
    load_q_var: NDArray[np.float64],
    der_idx: Sequence[int],
    der_p_w: NDArray[np.float64] | None,
    der_q_var: NDArray[np.float64] | None,
    source: str,
    v_src: float,
    i_neg: complex | None = None,
) -> None:
    """Set the customers' loads and the DER generators (PV, storage, DG; W, var), the source's
    voltage (pu) and, with ``i_neg`` (A, :func:`negative_sequence_current`), the current of its
    negative-sequence source in the engine; ``der_p_w`` None leaves the generators as they
    are."""
    for j, p, q in zip(load_idx, load_p_w, load_q_var, strict=True):
        api.Loads.Idx(j)
        api.Loads.kW(float(p) / 1e3)
        api.Loads.kvar(float(q) / 1e3)
    if der_p_w is not None and der_q_var is not None:
        for j, p, q in zip(der_idx, der_p_w, der_q_var, strict=True):
            api.Generators.Idx(j)
            api.Generators.kW(float(p) / 1e3)
            api.Generators.kvar(float(q) / 1e3)
    api.Vsources.Name(source)
    api.Vsources.PU(v_src)
    if i_neg is not None:
        api.Isource.Name(NEG_SEQUENCE_SOURCE)
        api.Isource.Amps(abs(i_neg))
        api.Isource.AngleDeg(float(np.degrees(np.angle(i_neg))))


def der_setpoints(
    pv_p_w: NDArray[np.float64] | None,
    pv_q_var: NDArray[np.float64] | None,
    bess_p_w: NDArray[np.float64] | None,
    dg_p_w: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """The DER generators' setpoints in :attr:`VariantRunner.der_generators` order (PV, then
    storage and DG at q = 0); None without units."""
    pv = [] if pv_p_w is None or pv_q_var is None else [(pv_p_w, pv_q_var)]
    fixed = [(p, np.zeros(p.size)) for p in (bess_p_w, dg_p_w) if p is not None]
    parts = pv + fixed
    if not parts:
        return None, None
    return np.concatenate([p for p, _ in parts]), np.concatenate([q for _, q in parts])


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioOutcome:
    scenario_id: int
    network_id: str
    variant_id: str
    seed: int
    """Fingerprint of the scenario's seed sequence."""
    pf_status: PFStatus
    state_id: str | None
    origin: Origin | None
    """How this scenario's controls reached its state (null with the state)."""
    profile_source: str
    profile_time_s: int | None
    pv_time_utc_s: int | None
    """Start of the PVGIS hour drawn (Unix time, s); None without PV or with the clear sky."""
    der_penetration: float
    source_v_pu: float
    load_p_w: NDArray[np.float64]
    load_q_var: NDArray[np.float64]
    """Setpoints of the customers (enabled loads ordered by name)."""
    pv_p_w: NDArray[np.float64] | None
    pv_q_var: NDArray[np.float64] | None
    pv_p_avail_w: NDArray[np.float64] | None
    bess_p_w: NDArray[np.float64] | None = None
    """Storage set-points (W, discharging positive; q = 0), in ``runner.storage.units`` order."""
    dg_p_w: NDArray[np.float64] | None = None
    """DG set-points (W; q = 0), in ``runner.dg.units`` order."""
    dg_p_avail_w: NDArray[np.float64] | None = None
    """DG available capacities ``a · P^max`` (W), the OPF's bound of their ``p`` and ``q``."""
    source_u2: float = 0.0
    """Negative-sequence ratio of the source's ideal voltage (:func:`scenario_e`)."""
    source_u2_angle_rad: float | None = None
    """Angle of its negative sequence relative to the positive one (rad); None when not drawn."""
    n_off_setpoint: int = 0
    pf_iterations: int = 0
    residual_rel: float | None = None
    t_solve_s: float = 0.0
    message: str | None = None
    v: NDArray[np.complex128] | None = None
    """Node voltages (V) in the model's node order; only when converged at the setpoints."""
    i_pd: NDArray[np.complex128] | None = None
    """Currents into the PD elements per local position (``VariantRunner.plan`` rows, A); only
    with ``labels.element_currents`` (otherwise ``Yprim · V``, recomputed when needed)."""
    t_extract_s: float = 0.0


class VariantRunner:
    """Runs the scenarios of one variant in one OpenDSS context."""

    def __init__(
        self,
        network: Network,
        variant: VariantResult,
        config: GenerationConfig,
        i: int,
        *,
        root: Path | None = None,
        prepared: PreparedNetwork | None = None,
    ) -> None:
        """``prepared``: the variant's compiled network (``generate_variant(keep_prepared=True)``),
        taken over with its context; None compiles it again."""
        if variant.perturbation is None:
            raise ValueError(f"{variant.variant_id} failed: no scenarios to run")
        self.config = config
        self.i, self.k = i, variant.variant_idx
        self.net = network_key(network.id)
        """Key of the network's draws (the index ``i`` only numbers its scenario ids)."""
        self.variant_id = variant.variant_id
        self.network_id = network.id
        self.options = prepare_options(config)
        if prepared is None:
            prepared = prepare_network(
                network, self.options, root, variant_id=variant.variant_id, edits=variant.script
            )
        elif (prepared.model.variant_id, prepared.options) != (variant.variant_id, self.options):
            raise ValueError(f"{variant.variant_id}: the compiled network is another variant's")
        self.ctx, self.model = prepared.ctx, prepared.model
        self.customers = customers_of(prepared)
        self.loads = make_provider(config.scenarios.profiles, self.customers, network, root)
        units = pv_units(variant.perturbation, config.der.pv)
        profile = config.scenarios.pv_profile
        self.pv = (
            PVOutputs(units, config.der.pv, profile, series_for(profile, network.id, root))
            if units
            else None
        )
        bess = storage_units(variant.perturbation)
        self.storage = StorageSetpoints(bess, config.scenarios.storage_profile) if bess else None
        dg = dg_units(variant.perturbation)
        self.dg = DGSetpoints(dg, config.scenarios.dg_profile) if dg else None
        names = [u.generator for u in units] + [u.generator for u in (*bess, *dg)]
        self.der_generators = tuple(names)
        """The DER generators: PV units, storage units, then DG (the rows of
        ``scenarios/der``)."""
        n = len(self.customers)
        self.der_penetration = len(units) / n if n else 0.0
        self.initial_state = read_state(self.ctx, self.variant_id, "as_is")
        self._settles = has_controls(self.ctx)
        """Without a control element the label solve is the power flow: no settling solves."""
        if not self._settles:
            self.ctx.command("Set ControlMode=OFF")
        self._load_idx = engine_indices(
            self.ctx.api.Loads.AllNames(), (c.element for c in self.customers)
        )
        self._der_idx = engine_indices(self.ctx.api.Generators.AllNames(), self.der_generators)
        """Engine indices (1-based) of the customers' loads and the DER generators."""
        self.registry = StateRegistry()
        self.plan = build_plan(self.model)
        """Rows of the element results (element, terminal, position)."""
        self.y_base = self.model.ybus()
        """Network admittance of the variant as compiled (S): ``variants/ybus``, the base of the
        entries stored per state."""
        self._states: dict[str, tuple[ElectricalModel, ExtractionPlan]] = {}
        self._met: set[str] = set()
        self._new: dict[str, ElectricalModel] = {}
        self._to_model = self.ctx.ynode_positions(n.opendss_node for n in self.model.nodes)
        self.ctx.api.Vsources.First()
        self._source = str(self.ctx.api.Vsources.Name())
        self.unbalance = (
            add_negative_sequence_source(self.ctx, self.model)
            if config.scenarios.source_unbalance.support()[1] > 0
            else None
        )
        """The source, when the circuit holds its negative-sequence current source."""
        self._draws: _Draws | None = None
        """The current scenario's draws (the load-ramp fallback scales them)."""
        self._note: str | None = None
        """Why the current scenario's controls did not settle, when the fallback was tried."""

    def run(self, s: int) -> ScenarioOutcome:
        """Scenario ``s`` of the variant."""
        seed, net, k = self.config.settings.seed, self.net, self.k
        start = time.perf_counter()
        draws = _Draws(np.zeros(0), np.zeros(0), None, None, 0.0, None)
        try:
            loads = self.loads.draw(seed, net, k, s)
            pv = self.pv.draw(seed, net, k, s, loads.time_s) if self.pv is not None else None
            rng = scenario_rng(seed, net, k, s, ScenarioStream.SOURCE_VOLTAGE)
            v_src = float(self.config.scenarios.source_voltage_pu.sample(rng))
            u2, angle = 0.0, None
            if self.unbalance is not None:
                rng = scenario_rng(seed, net, k, s, ScenarioStream.SOURCE_UNBALANCE)
                u2 = float(self.config.scenarios.source_unbalance.sample(rng))
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
            bess = self.storage.draw(seed, net, k, s) if self.storage is not None else None
            dg = self.dg.draw(seed, net, k, s) if self.dg is not None else None
            dg_avail = self.dg.capacity(seed, net, k, s) if self.dg is not None else None
            draws = _Draws(loads.p_w, loads.q_var, pv, loads.time_s, v_src, bess, dg, dg_avail,
                           u2, angle)  # fmt: skip
            self._set(draws)
            self._draws, self._note = draws, None
            state, status = self._controls(s)
            if status is not None and not status.converged:
                return self._outcome(
                    s, draws, start, "not_converged", None, status.iterations, note=self._note
                )
            status = self._solve()
            self._at_state(state.state_id)  # the state's Yprims, rebuilt by this solve
            state_id = self.registry.add(state)  # once it reaches the dataset
            if not status.converged:
                return self._outcome(s, draws, start, "not_converged", state, status.iterations)
            report = setpoint_report(self.ctx)
            if not report.passes():
                return self._outcome(
                    s, draws, start, "off_setpoint", state, status.iterations,
                    n_off=len(report.off),
                )  # fmt: skip
            solved = time.perf_counter()
            v = self.ctx.ynode_voltages()[self._to_model]
            model, plan = self._at_state(state_id)
            i_pd = plan.currents(v) if self.config.labels.element_currents else None
            extracted = time.perf_counter()
            return self._outcome(
                s, draws, start, "converged", state, status.iterations,
                rel=self._residual(s, model), v=v, i_pd=i_pd,
                t_solve_s=solved - start, t_extract_s=extracted - solved,
            )  # fmt: skip
        except Exception as exc:  # recorded, never dropped
            return self._outcome(s, draws, start, "error", None, 0, error=exc)

    # ------------------------------------------------------------------ steps

    def _set(self, draws: _Draws) -> None:
        pv = draws.pv
        der_p, der_q = der_setpoints(
            pv.p_w if pv is not None else None,
            pv.q_var if pv is not None else None,
            draws.bess,
            draws.dg,
        )
        i_neg = None
        if self.unbalance is not None and draws.u2_angle is not None:
            i_neg = negative_sequence_current(self.unbalance, draws.v_src, draws.u2, draws.u2_angle)
        set_setpoints(
            self.ctx.api, self._load_idx, draws.p_w, draws.q_var, self._der_idx, der_p, der_q,
            self._source, draws.v_src, i_neg,
        )  # fmt: skip

    def _at_state(self, state_id: str) -> tuple[ElectricalModel, ExtractionPlan]:
        """The model and plan at the current (frozen, solved) state, kept for the last states; a
        state met for the first time since :meth:`begin_unit` is also kept for
        :meth:`new_states`."""
        if state_id not in self._states:
            if len(self._states) >= MAX_CACHED_STATES:
                self._states.pop(next(iter(self._states)))
            model = model_at_state(self.model, self.ctx)
            self._states[state_id] = (model, self.plan.at_state(model))
        if state_id not in self._met:
            self._met.add(state_id)
            self._new[state_id] = self._states[state_id][0]
        return self._states[state_id]

    def state_model(self, state_id: str) -> ElectricalModel:
        """The model at a state met by a recent scenario (the last one's is always kept)."""
        return self._states[state_id][0]

    def begin_unit(self) -> None:
        """Start a work unit: :meth:`new_states` then reports every state the unit meets, also
        those met by earlier units of this runner (a unit's files do not depend on scheduling); the
        registry restarts with the unit, so it does not grow with the runner's history."""
        self.registry = StateRegistry()
        self._met = set()
        self._new = {}

    def new_states(self) -> dict[str, tuple[NetworkState, ElectricalModel]]:
        """States met since the last call or :meth:`begin_unit` (for the writer), with their
        model."""
        out = {sid: (self.registry.states[sid], model) for sid, model in self._new.items()}
        self._new = {}
        return out

    def _residual(self, s: int, model: ElectricalModel) -> float | None:
        sid = scenario_id(self.i, self.k, s)
        if not check_this_scenario(sid, self.config.settings.residual_sample_rate):
            return None
        v = self.ctx.ynode_voltages()[self._to_model]
        i_pc = injections(self.ctx, model)
        return residual(model, v, i_pc, tolerance=self.options.tolerance).rel

    def _controls(self, s: int) -> tuple[NetworkState, SolveStatus | None]:
        controls, seed = self.config.controls, self.config.settings.seed
        if not self._settles:
            origin: Origin = "as_is" if controls.mode == "as_is" else "static_settle"
            return replace(self.initial_state, origin=origin), None
        apply_state(self.ctx, self.initial_state)  # every scenario settles from the same state
        self.ctx.command("Set Mode=Snap")
        rng = scenario_rng(seed, self.net, self.k, s, ScenarioStream.TAP_JITTER)
        args = (self.ctx, controls.mode, self.variant_id)
        state, status = settle(*args, jitter_steps=controls.tap_jitter_steps, rng=rng)
        if status is not None and not status.converged:
            with self._more_iterations():
                apply_state(self.ctx, self.initial_state)
                self.ctx.command("Set Mode=Snap")
                rng = scenario_rng(seed, self.net, self.k, s, ScenarioStream.TAP_JITTER)
                state, status = settle(*args, jitter_steps=controls.tap_jitter_steps, rng=rng)
        if status is not None and not status.converged:
            return self._ramp(s)
        return state, status

    def _ramp(self, s: int) -> tuple[NetworkState, SolveStatus | None]:
        """Fallback when settling fails at the initial taps: OpenDSS moves the controls only after a
        converged solve, so a load the initial taps cannot carry never lets them move. The
        scenario's setpoints are raised from the initial state in :data:`RAMP_STEPS` equal steps
        with the controls active, the taps carried from step to step; then the controls settle
        as usual from the taps reached. Only the scenario's own draws enter: order-independent.
        A failure sets ``self._note`` (the scenario's ``pf_message``)."""
        controls, seed, ctx = self.config.controls, self.config.settings.seed, self.ctx
        draws = self._draws
        assert draws is not None
        apply_state(ctx, self.initial_state)
        ctx.command("Set Mode=Snap")
        ctx.command(f"Set ControlMode=STATIC MaxControlIter={MAX_CONTROL_ITER}")
        for step in range(1, RAMP_STEPS + 1):
            self._set(_scaled(draws, step / RAMP_STEPS))
            step_status = ctx.solve()
            if not step_status.converged:
                ctx.command("Set ControlMode=OFF")
                self._set(draws)
                self._note = (
                    "settling failed at the initial taps; load ramp stopped at "
                    f"{step}/{RAMP_STEPS} of the setpoints"
                )
                return self.initial_state, step_status
        rng = scenario_rng(seed, self.net, self.k, s, ScenarioStream.TAP_JITTER)
        args = (ctx, controls.mode, self.variant_id)
        state, status = settle(*args, jitter_steps=controls.tap_jitter_steps, rng=rng)
        if status is not None and not status.converged:
            self._note = "settling failed at the initial taps; after the load ramp too"
        ramped: Origin = (
            "static_settle+ramp+jitter"
            if state.origin == "static_settle+jitter"
            else "static_settle+ramp"
        )
        return replace(state, origin=ramped), status

    def _solve(self) -> SolveStatus:
        self.ctx.command("Set Mode=Snap")
        status = self.ctx.solve()
        if status.converged:
            return status
        with self._more_iterations():
            self.ctx.command("Set Mode=Snap")
            return self.ctx.solve()

    def _more_iterations(self) -> _Iterations:
        return _Iterations(self.ctx, self.options.max_iterations)

    def _outcome(
        self,
        s: int,
        draws: _Draws,
        start: float,
        status: PFStatus,
        state: NetworkState | None,
        iterations: int,
        *,
        n_off: int = 0,
        rel: float | None = None,
        v: NDArray[np.complex128] | None = None,
        i_pd: NDArray[np.complex128] | None = None,
        t_solve_s: float | None = None,
        t_extract_s: float = 0.0,
        error: Exception | None = None,
        note: str | None = None,
    ) -> ScenarioOutcome:
        message = note
        if error is not None:
            first = str(error).splitlines()[0] if str(error) else ""
            message = f"{type(error).__name__}: {first}"
        pv = draws.pv
        return ScenarioOutcome(
            scenario_id=scenario_id(self.i, self.k, s),
            network_id=self.network_id,
            variant_id=self.variant_id,
            seed=scenario_seed(self.config.settings.seed, self.net, self.k, s),
            pf_status=status,
            state_id=None if state is None else state.state_id,
            origin=None if state is None else state.origin,
            profile_source=self.loads.name,
            profile_time_s=draws.time_s,
            pv_time_utc_s=None if pv is None else pv.time_utc_s,
            der_penetration=self.der_penetration,
            source_v_pu=draws.v_src,
            load_p_w=draws.p_w,
            load_q_var=draws.q_var,
            pv_p_w=None if pv is None else pv.p_w,
            pv_q_var=None if pv is None else pv.q_var,
            pv_p_avail_w=None if pv is None else pv.p_avail_w,
            bess_p_w=draws.bess,
            dg_p_w=draws.dg,
            dg_p_avail_w=draws.dg_avail,
            source_u2=draws.u2,
            source_u2_angle_rad=draws.u2_angle,
            n_off_setpoint=n_off,
            pf_iterations=iterations,
            residual_rel=rel,
            t_solve_s=time.perf_counter() - start if t_solve_s is None else t_solve_s,
            message=message,
            v=v,
            i_pd=i_pd,
            t_extract_s=t_extract_s,
        )


@dataclass(frozen=True, slots=True)
class _Draws:
    p_w: NDArray[np.float64]
    q_var: NDArray[np.float64]
    pv: PVDraw | None
    time_s: int | None
    v_src: float
    bess: NDArray[np.float64] | None
    """Storage set-points (W)."""
    dg: NDArray[np.float64] | None = None
    """DG set-points (W)."""
    dg_avail: NDArray[np.float64] | None = None
    """DG available capacities (W), not ramped (as PV's available power)."""
    u2: float = 0.0
    """Negative-sequence ratio of the source."""
    u2_angle: float | None = None
    """Its angle (rad); None when not drawn."""


def _scaled(draws: _Draws, factor: float) -> _Draws:
    """``draws`` with every load, PV, storage and DG setpoint times ``factor`` (source voltage and
    unbalance unchanged)."""
    pv = draws.pv
    if pv is not None:
        pv = replace(pv, p_w=factor * pv.p_w, q_var=factor * pv.q_var)
    bess = None if draws.bess is None else factor * draws.bess
    dg = None if draws.dg is None else factor * draws.dg
    return replace(
        draws, p_w=factor * draws.p_w, q_var=factor * draws.q_var, pv=pv, bess=bess, dg=dg
    )


class _Iterations:
    """``MaxIterations`` x RETRY_FACTOR inside the block."""

    def __init__(self, ctx: DssContext, base: int) -> None:
        self.ctx, self.base = ctx, base

    def __enter__(self) -> None:
        self.ctx.command(f"Set MaxIterations={self.base * RETRY_FACTOR}")

    def __exit__(self, *exc: object) -> None:
        self.ctx.command(f"Set MaxIterations={self.base}")
