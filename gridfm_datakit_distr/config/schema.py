"""Generation config models.

A config is a YAML file (``scripts/config/*.yaml``) read by
:func:`gridfm_datakit_distr.config.load_config`.
The models forbid unknown keys and non-finite numbers and do not coerce types (``"1"`` is not an
integer). Value specs (``value_spec.py``, gridfm-datakit's semantics) are parsed with the range
their quantity admits. The model docstrings and field descriptions are the user reference:
``config/reference.py`` renders them as docs/manual/config.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    model_validator,
)

from gridfm_datakit_distr.config.value_spec import (
    NON_NEGATIVE,
    POSITIVE,
    Fixed,
    Floor,
    Uniform,
    ValueSpec,
    parse_spec,
)
from gridfm_datakit_distr.model.ids import MAX_SCENARIO_INDEX, MAX_VARIANT_INDEX

MAX_VARIANTS = MAX_VARIANT_INDEX + 1
"""Variant ids keep four digits (``v{k:04d}``)."""
MAX_SCENARIOS_PER_VARIANT = MAX_SCENARIO_INDEX + 1
"""Scenario ids hold 31 bits of scenario index."""
MAX_TOLERANCE = 1e-8
"""The setpoint check needs ``Tolerance`` ≤ its relative tolerance (1e-6) / 100."""

VOLTAGE_PRESETS: dict[str, tuple[float, float]] = {
    "uk_esqcr": (0.94, 1.10),
    "ansi_a": (0.95, 1.05),
}
"""Voltage limits (pu): UK ESQCR 2002, 230 V +10 %/-6 % (216.2 to 253.0 V) at LV supply
terminals; ANSI C84.1 Range A service voltage, +/-5 %."""


@dataclass(frozen=True, slots=True)
class SpecRange:
    """Values a value-spec field admits: a floor (value-spec semantics) and a ceiling that the
    spec's whole support must respect (a normal needs an explicit ``max`` then)."""

    floor: Floor | None = None
    ceiling: float | None = None

    def parse(self, value: object) -> ValueSpec:
        spec = value if isinstance(value, ValueSpec) else parse_spec(value, self.floor)
        if self.ceiling is not None:
            high = spec.support()[1]
            if high > self.ceiling:
                raise ValueError(
                    f"every value must be <= {self.ceiling:g}, but draws reach {high:g}"
                )
        return spec

    def describe(self) -> str:
        parts = []
        if self.floor is not None:
            parts.append(f"{'>' if self.floor.strict else '≥'} {self.floor.bound:g}")
        if self.ceiling is not None:
            parts.append(f"≤ {self.ceiling:g}")
        return "value spec" + (", " + " and ".join(parts) if parts else "")


@dataclass(frozen=True, slots=True)
class DocType:
    """Type label for the reference doc of a field validated by a custom validator."""

    text: str


def _dump_spec(spec: ValueSpec) -> Any:
    return spec.to_config()


_POSITIVE = SpecRange(POSITIVE)
_NON_NEGATIVE = SpecRange(NON_NEGATIVE)
_FRACTION = SpecRange(NON_NEGATIVE, 1.0)
_POWER_FACTOR = SpecRange(POSITIVE, 1.0)

PositiveSpec = Annotated[
    ValueSpec, _POSITIVE, PlainValidator(_POSITIVE.parse), PlainSerializer(_dump_spec)
]
NonNegativeSpec = Annotated[
    ValueSpec, _NON_NEGATIVE, PlainValidator(_NON_NEGATIVE.parse), PlainSerializer(_dump_spec)
]
FractionSpec = Annotated[
    ValueSpec, _FRACTION, PlainValidator(_FRACTION.parse), PlainSerializer(_dump_spec)
]
PowerFactorSpec = Annotated[
    ValueSpec, _POWER_FACTOR, PlainValidator(_POWER_FACTOR.parse), PlainSerializer(_dump_spec)
]


def _s_base(value: object) -> str | float:
    if value == "auto":
        return "auto"
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        return float(value)
    raise ValueError(f"must be 'auto' or a number > 0 (VA), got {value!r}")


SBase = Annotated[
    Literal["auto"] | float,
    DocType("`auto` or a number > 0 (VA)"),
    PlainValidator(_s_base),
]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class VoltageLimits(_Config):
    """Voltage limits, per unit of each bus's nominal line-to-neutral voltage
    (`variants/nodes.kv_nominal_ln`: the base unless the catalog declares a nominal, e.g. 230 V on
    the 416 V CSIRO feeders). They do not change the power-flow labels: they bound the OPF, count
    the violations in `stats` and set `min_vm_pu`, `max_vm_pu` of `export-posseq`. The defaults
    are the `uk_esqcr` values."""

    preset: Literal["uk_esqcr", "ansi_a"] | None = Field(
        default=None,
        description="`uk_esqcr`: 0.94 to 1.10 pu (UK ESQCR, 230 V +10 %/-6 %); `ansi_a`: "
        "0.95 to 1.05 pu (ANSI C84.1 Range A). Sets `vmin_pu` and `vmax_pu`: giving either "
        "with another value is an error. null: `vmin_pu` and `vmax_pu` as given.",
    )
    vmin_pu: float = Field(
        default=0.94,
        gt=0,
        description="Lower limit (pu of the nominal voltage); must be below `vmax_pu`.",
    )
    vmax_pu: float = Field(
        default=1.10, gt=0, description="Upper limit (pu of the nominal voltage)."
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("preset") not in VOLTAGE_PRESETS:
            return data
        preset = data["preset"]
        vmin, vmax = VOLTAGE_PRESETS[preset]
        given = {k: data[k] for k in ("vmin_pu", "vmax_pu") if k in data}
        differ = {k: v for k, v in given.items() if v != {"vmin_pu": vmin, "vmax_pu": vmax}[k]}
        if differ:
            got = ", ".join(f"{k} {v}" for k, v in differ.items())
            raise ValueError(f"preset {preset!r} sets vmin_pu {vmin} and vmax_pu {vmax}; got {got}")
        return {**data, "vmin_pu": vmin, "vmax_pu": vmax}

    @model_validator(mode="after")
    def _ordered(self) -> VoltageLimits:
        if self.vmin_pu >= self.vmax_pu:
            raise ValueError(
                f"vmin_pu must be below vmax_pu, got {self.vmin_pu} and {self.vmax_pu}"
            )
        return self


class Networks(_Config):
    """Which catalog networks enter the dataset and which model of each is loaded
    ([Networks](networks.md))."""

    include: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        description="Catalog network ids or glob patterns (`fnmatch`, case-sensitive; `*` also "
        "matches `/`), e.g. `ieee13`, `csiro_enwl/n01_*`, `*`. Each entry must match a network; "
        "overlapping matches count once. `<pattern>@<representation>` loads that alternative "
        "model of every network the pattern matches (an error where one lacks it or is already "
        "selected with another representation) and wins over a plain match of the same network: "
        "`four_wire` (the default), `kron`, "
        "`phase_to_neutral`, `modified_phase_to_neutral` on the CSIRO feeders, `unbalanced` on "
        "`ieee8500`; the alternatives are not yet validated. Patterns do not skip networks that "
        "fail the conformance suite: `*` selects `csiro_enwl/n13_f4`, whose variants do not "
        "converge. The selected networks are sorted by id: that order numbers them in scenario "
        "ids and file names (`network_idx`); a network's random draws depend on its id only.",
    )
    neutral_representation: Literal["explicit", "kron", "as_is"] = Field(
        default="explicit",
        description="`kron`: the `kron` representation of every network that has one (the CSIRO "
        "feeders), the default model elsewhere. `explicit` and `as_is` both load each network's "
        "default model; `explicit` also has `generate` print a note naming the networks whose "
        "default model has no explicit neutral (the Kron-reduced IEEE feeders). An "
        "`@representation` in `include` overrides this key.",
    )
    voltage_limits: VoltageLimits = VoltageLimits()


class LoadNormalisation(_Config):
    """Model of every enabled load. Outside its voltage window [`vminpu`, `vmaxpu`] OpenDSS turns
    a load into a constant impedance and still reports convergence; every solve is checked against
    the setpoints, so a scenario where a load or generator misses its setpoint (1e-6 relative) is
    `off_setpoint`, without results. The PV units get the same window."""

    model: Literal["constant_power"] = Field(
        default="constant_power",
        description="`constant_power`, the only model: every load becomes constant P, Q "
        "(OpenDSS `model=1`), whatever its model in the source.",
    )
    vminpu: float = Field(
        default=0.6,
        gt=0,
        lt=1,
        description="Lower end of the window (pu of each load's rated voltage). OpenDSS's own "
        "0.95 would turn loads into impedances in ordinary undervoltage.",
    )
    vmaxpu: float = Field(
        default=1.5,
        gt=1,
        description="Upper end of the window (pu of each load's rated voltage).",
    )


class Normalisation(_Config):
    """Normalisation applied to every network and variant after compilation, recorded in the
    passports (`catalog/passports`)."""

    loads: LoadNormalisation = LoadNormalisation()
    s_base_va: SBase = Field(
        default="auto",
        description="Power base (VA) of each network: the per-unit base of the semantic model "
        "(`variants/semantic`, PMD-aligned), `catalog/networks.s_base_va` and the passports, "
        "and in `export-posseq` the source generator's limits and the balance tolerance of the "
        "positive-sequence view (itself per unit on 100 MVA). `auto`: per network, the rating "
        "of the largest transformer with a winding in the source zone, else the total nominal "
        "apparent power of the loads, else 1 MVA. A number: that base for every network.",
    )


class Admittance(_Config):
    """Parameter perturbations, drawn per element and variant. A fixed `1.0` (null for the
    resistance) keeps the source's values and draws nothing."""

    line_length_scale: PositiveSpec = Field(
        default=Uniform(0.9, 1.1),
        description="Factor on each line's length (switches excluded): its series impedance and "
        "shunt admittance scale with it.",
    )
    neutral_impedance_scale: PositiveSpec = Field(
        default=Fixed(1.0),
        description="Factor β on the neutral of every line with a neutral conductor: its self "
        "impedance times β, the phase-neutral mutual impedances times √β. No effect on networks "
        "without an explicit neutral.",
    )
    grounding_resistance_ohm: PositiveSpec | None = Field(
        default=None,
        description="Resistance (Ω) of every earth electrode: `Rneut` of grounded wye windings "
        "and grounding reactors, except the neutral's bond at a source. null: the source's "
        "values.",
    )


class PhaseReallocation(_Config):
    """Moves single-phase loads (one phase conductor, at a bus with another phase) to another
    phase of their bus, with their neutral; phase-to-phase loads stay. Drawn per variant."""

    enabled: bool = Field(
        default=True, description="Reallocate in each variant; `false` ignores `fraction`."
    )
    fraction: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Share of the eligible loads moved (rounded to the nearest count), each to "
        "one of the other phases of its bus, chosen uniformly.",
    )


class Variants(_Config):
    """Perturbed copies of each network ([Variants](variants.md)). Each variant draws from its own
    streams; its base case must converge and pass the Kirchhoff residual check, else it is redrawn,
    3 attempts at most, and then recorded as failed: its units fail and `generate` exits with an
    error. The defaults perturb every variant; `line_length_scale: 1.0`,
    `phase_reallocation.enabled: false` and `der.pv.penetration: 0` keep the source network."""

    n_variants: int = Field(
        default=4,
        ge=1,
        le=MAX_VARIANTS,
        description="Variants per network (ids `v0000` to `v9999`).",
    )
    admittance: Admittance = Admittance()
    phase_reallocation: PhaseReallocation = PhaseReallocation()


class PV(_Config):
    """Rooftop PV units, placed and sized per variant (`variants/variants.perturbation`), output
    per scenario (`scenarios.pv_profile`). Every enabled load is a customer that can get one: the
    unit sits on the load's nodes (after any phase move), constant P, Q within the load window."""

    penetration: FractionSpec = Field(
        default=Uniform(0.0, 0.8),
        description="Share of customers with PV, drawn per variant: each customer gets a unit "
        "with that probability, so the realised share (`scenario_meta.der_penetration`) varies "
        "around the draw. `0`: no PV. The default puts a voltage violation in about 6 % of the "
        "scenarios of a dataset of the whole catalog.",
    )
    kwp: PositiveSpec = Field(
        default=Uniform(2.0, 6.0),
        description="Peak power (kW) of each unit, drawn per customer and variant.",
    )
    power_factor: PowerFactorSpec = Field(
        default=Fixed(1.0),
        description="Power factor of each inverter in the power flow, drawn per unit and "
        "variant. Below 1 the inverter absorbs reactive power, `q = -p · tan(acos(pf))`, limited "
        "to its rating.",
    )
    inverter_kva_per_kwp: float = Field(
        default=1.1,
        gt=0,
        description="Inverter rating (kVA) per kWp. Active power is clipped at the rating; from "
        "1 up it never clips (the available power never exceeds kWp).",
    )
    reactive_fraction: FractionSpec = Field(
        default=Fixed(0.44),
        description="OPF only: largest reactive power of each inverter, as a fraction of its "
        "rating (IEEE 1547-2018 Category B: 0.44; 1: the rating's circle only); drawn per unit "
        "and variant.",
    )
    reactive_threshold: FractionSpec = Field(
        default=Fixed(0.05),
        description="OPF only: a unit whose available power is below this fraction of its "
        "rating gives no reactive power (IEEE 1547-2018 Category B: 0.05; 0: no threshold); "
        "drawn per unit and variant.",
    )


class DER(_Config):
    """Distributed energy resources added to every variant."""

    pv: PV = PV()


class SyntheticProfiles(_Config):
    """Synthetic loads, `provider: synthetic` only (this section with another provider is an
    error): `P = g · m · P_nominal` and lagging `Q = P · tan(acos(pf))`, with `g` per scenario, `m`
    and `pf` per customer and scenario. The loads' nominal reactive powers are not used."""

    global_scale: NonNegativeSpec = Field(
        default=Uniform(0.2, 1.2),
        description="Factor `g` on every nominal load, drawn per scenario.",
    )
    local_sigma: float = Field(
        default=0.3,
        ge=0,
        description="Sigma of the per-customer factor `m = exp(sigma z - sigma²/2)`, `z` standard "
        "normal, of mean 1 (so the mean load is `E[g] · P_nominal`); 0: no per-customer noise.",
    )
    power_factor: PowerFactorSpec = Field(
        default=Uniform(0.9, 0.99),
        description="Power factor `pf` of each load, drawn per customer and scenario.",
    )


class Profiles(_Config):
    """Where each scenario's loads come from ([Scenarios](scenarios.md))."""

    provider: Literal["synthetic", "network_loadshapes", "edp_smart_meter"] = Field(
        default="synthetic",
        description="`synthetic`: nominal loads times random factors (section `synthetic`). "
        "`network_loadshapes`: the loads' daily (else yearly) loadshapes in the source model, at "
        "a profile time drawn uniformly over the shortest multi-point loadshape; loads without "
        "one keep their nominal (or one-point) value. `edp_smart_meter`: the same mechanism for "
        "the EDP networks, whose loadshapes are the smart-meter files; only the recorded "
        "`profile_source` differs. `generate` refuses either on a network without loadshapes "
        "before any work, also with `--dry-run`.",
    )
    same_timestamp: bool = Field(
        default=True,
        description="Loadshape providers only (ignored by `synthetic`). `true`: one profile time "
        "per scenario for all customers (coincident demand), whose time of day is the PV hour "
        "(the profile start taken as midnight). `false`: one time per customer, and a PV hour "
        "drawn on its own.",
    )
    synthetic: SyntheticProfiles = SyntheticProfiles()

    @model_validator(mode="after")
    def _synthetic_needs_its_provider(self) -> Profiles:
        if self.provider != "synthetic" and "synthetic" in self.model_fields_set:
            raise ValueError(
                f"synthetic is used only with provider 'synthetic', got provider {self.provider!r}"
            )
        return self


class PVProfile(_Config):
    """PV output per scenario: available power = kWp · clear-sky shape of the hour · cloud
    factor, injected up to the inverter rating ([Scenarios](scenarios.md))."""

    provider: Literal["synthetic_clear_sky"] = Field(
        default="synthetic_clear_sky",
        description="`synthetic_clear_sky`: `max(0, cos(π (h - 12) / 12))` between 6:00 and "
        "18:00, no date or location. The hour `h` is the load profile's time of day when the "
        "scenario has one, else uniform over 24 h: with `synthetic` loads half the scenarios "
        "fall at night, with no PV output.",
    )
    cloudiness: FractionSpec = Field(
        default=Uniform(0.0, 1.0),
        description="Cloud cover fraction `c`, drawn per scenario for the whole network; the "
        "clear-sky output is scaled by `1 - 0.75 c^3.4` (Kasten-Czeplak): 1 leaves 25 %.",
    )


class Scenarios(_Config):
    """Operating points of each variant ([Scenarios](scenarios.md)); each scenario draws from its
    own streams."""

    n_scenarios_per_variant: int = Field(
        ge=1,
        le=MAX_SCENARIOS_PER_VARIANT,
        description="Scenarios per variant of each network.",
    )
    profiles: Profiles = Profiles()
    pv_profile: PVProfile = PVProfile()
    source_voltage_pu: PositiveSpec = Field(
        default=Uniform(0.98, 1.05),
        description="Source voltage magnitude (pu of the source's base voltage), drawn per "
        "scenario; balanced, angle 0.",
    )


class Controls(_Config):
    """Regulator taps and switched-capacitor states per scenario. The state reached is frozen
    (`states/`) and the label is the power flow at it with `ControlMode=OFF`. On a network without
    control elements both keys are ignored: the state is the compiled one."""

    mode: Literal["static_settle", "as_is"] = Field(
        default="static_settle",
        description="`static_settle`: with the scenario's setpoints, `ControlMode=STATIC` "
        "settles regulators and capacitors, the regulated taps are jittered and settle again "
        "from there, and the state is frozen; when settling fails, a load ramp reaches the "
        "scenario ([Scenarios](scenarios.md)). `as_is`: the taps and capacitor states as "
        "compiled.",
    )
    tap_jitter_steps: int = Field(
        default=2,
        ge=0,
        description="Largest jitter `J` (tap steps): each regulated winding's settled tap moves "
        "by a uniform integer in [-J, J], within its limits, before the controls settle again; "
        "capacitors are not jittered. 0: settle once. Ignored with `as_is`.",
    )


class SampledCosts(_Config):
    """Cost coefficients of the preset ``sampled``, drawn per scenario (``c2 p^2 + c1 p``, ``p``
    in W): one pair for the source, one per PV unit. Ignored with ``costs: economic``."""

    source_c1: NonNegativeSpec = Field(
        default=Fixed(1.0), description="Linear cost of the source's power (per W)."
    )
    source_c2: NonNegativeSpec = Field(
        default=Fixed(0.0), description="Quadratic cost of the source's power (per W²)."
    )
    pv_c1: NonNegativeSpec = Field(
        default=Uniform(0.0, 1.0),
        description="Linear cost of each PV unit's power (per W), drawn per unit.",
    )
    pv_c2: NonNegativeSpec = Field(
        default=Fixed(0.0), description="Quadratic cost of each PV unit's power (per W²)."
    )


class OPFLabels(_Config):
    """OPF labels: the minimum-cost OPF of every scenario, verified in OpenDSS
    ([OPF labels](opf.md)). The power-flow labels stay; the `opf/` tables are added."""

    enabled: bool = Field(
        default=False,
        description="Solve and verify the OPF of every scenario. The config is refused unless the "
        "`opf` extra (CasADi) is installed. Networks outside the OPF's scope get `not_supported` "
        "rows.",
    )
    costs: Literal["economic", "sampled"] = Field(
        default="economic",
        description="Cost preset. `economic`: source 1 per W, PV units 0 (minimum import); "
        "`sampled`: coefficients drawn per scenario from `sampled`.",
    )
    sampled: SampledCosts = SampledCosts()
    time_limit_s: float = Field(
        default=300.0,
        gt=0,
        description="Wall-clock limit (s) of each Ipopt solve (`max_wall_time`). A scenario not "
        "solved from the flat start is solved once more from its power flow, so it can take "
        "twice this. Machine-dependent; the 3000-iteration limit is the deterministic bound.",
    )


class Labels(_Config):
    """What is solved and stored for each scenario. The power-flow labels are always computed."""

    element_currents: bool = Field(
        default=False,
        description="Also store the current of every power-delivery element at each terminal "
        "conductor (`scenarios/element_results`); otherwise they follow exactly from the stored "
        "data as Yprim · V ([Outputs](outputs.md#6-scenario-level)).",
    )
    opf: OPFLabels = OPFLabels()


class Views(_Config):
    """Reduced views written next to the full model ([Outputs](outputs.md#7-views)). The
    positive-sequence view is not one: ``export-posseq`` computes it from a dataset."""

    kron: bool = Field(
        default=False,
        description="Kron view (neutral eliminated), re-solved in OpenDSS for every converged "
        "scenario (one more solve each; `views/kron/`); ~1 kB per 100 phase nodes and scenario. "
        "On a network without an explicit neutral it is only a mapping to the full model.",
    )


class Settings(_Config):
    """How the run is executed and written. `data_dir`, `num_processes`, `contexts_per_process`
    and `overwrite` are execution-only: the dataset does not depend on them, they are outside the
    config hash, and `generate` overrides them with `--data-dir`, `--processes`, `--contexts` and
    `--overwrite`."""

    data_dir: str = Field(
        default="./data",
        min_length=1,
        description="Parent directory of the dataset, `<data_dir>/<dataset_name>`; a relative "
        "path is relative to the working directory.",
    )
    seed: int = Field(
        ge=0,
        description="Root of the seed tree. Every draw is keyed by the seed, the network id, the "
        "variant and scenario indices and the quantity, never by the order of work: on one "
        "machine the same config, seed and versions give the same tables, but for the measured "
        "solve times, for any number of processes and contexts ([Scenarios](scenarios.md), "
        "Reproducibility in [OPF labels](opf.md)); a network's draws do not depend on the other "
        "networks selected; more variants or scenarios keep the existing ones; changing one "
        "quantity's spec does not shift the draws of another.",
    )
    num_processes: int | None = Field(
        default=None,
        ge=1,
        description="Worker processes; null: one per CPU core (`os.cpu_count()`); 1: in the "
        "main process.",
    )
    contexts_per_process: int = Field(
        default=1,
        ge=1,
        description="Threads per process, each with its own OpenDSS context and one chunk of the "
        "same variant: more than 1 helps only when a variant has several chunks "
        "(`n_scenarios_per_variant` > `chunk_size`).",
    )
    chunk_size: int = Field(
        default=500,
        ge=1,
        description="Scenarios per work unit: one Parquet part per scenario table, the unit of "
        "scheduling, resume and failure, held in memory until written.",
    )
    float_precision: Literal["mixed", "full"] = Field(
        default="mixed",
        description="`mixed`: float32 for currents, powers and setpoints, float64 for voltages "
        "and admittances; `full`: float64 everywhere ([Outputs](outputs.md#1-conventions)).",
    )
    residual_sample_rate: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="Share of converged scenarios whose full Kirchhoff (KCL) residual is computed "
        "and stored in `scenario_meta.residual_rel` (null elsewhere), chosen by a hash of the "
        "scenario id; it never changes a status.",
    )
    overwrite: bool = Field(
        default=False,
        description="Delete and regenerate the dataset in `<data_dir>/<dataset_name>` when it "
        "has another config hash or versions; `false` refuses. A directory that is not a "
        "dataset (no `manifest.json`) is never touched.",
    )
    validation_tolerance: float = Field(
        default=1e-10,
        gt=0,
        le=MAX_TOLERANCE,
        description="OpenDSS `Tolerance` of every solve of the generation, also the solver term "
        "of the residual check. At most 1e-8: the setpoint check (1e-6 relative) needs a "
        "tolerance 100 times tighter.",
    )


class GenerationConfig(_Config):
    """Top-level keys. `networks`, `scenarios` and `settings` hold required keys; the other
    sections may be omitted."""

    dataset_name: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Name of the dataset: its directory is `<settings.data_dir>/<dataset_name>`.",
    )
    networks: Networks
    normalisation: Normalisation = Normalisation()
    variants: Variants = Variants()
    der: DER = DER()
    scenarios: Scenarios
    controls: Controls = Controls()
    labels: Labels = Labels()
    views: Views = Views()
    settings: Settings
