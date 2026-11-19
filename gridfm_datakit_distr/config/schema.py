"""Generation config models.

A config is a YAML file (``scripts/config/*.yaml``) read by
:func:`gridfm_datakit_distr.config.load_config`.
The models forbid unknown keys and non-finite numbers and do not coerce types (``"1"`` is not an
integer). Value specs (``value_spec.py``, gridfm-datakit's semantics) are parsed with the range
their quantity admits. Defaults leave the source networks unchanged (no perturbation, no DER);
scenario sampling defaults are documented in docs/manual/config.md.
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
    """Voltage limits for statistics and OPF: a preset or explicit values, per unit of each
    network's nominal line-to-neutral voltage (the base unless the catalog declares one)."""

    preset: Literal["uk_esqcr", "ansi_a"] | None = Field(
        default=None,
        description="`uk_esqcr`: 0.94 to 1.10 pu (UK ESQCR, 230 V +10 %/-6 %); `ansi_a`: "
        "0.95 to 1.05 pu (ANSI C84.1 Range A). Sets `vmin_pu` and `vmax_pu`.",
    )
    vmin_pu: float = Field(
        default=0.94, gt=0, description="Lower voltage limit (pu of the nominal voltage)."
    )
    vmax_pu: float = Field(
        default=1.10, gt=0, description="Upper voltage limit (pu of the nominal voltage)."
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
    """Which catalog networks enter the dataset (`docs/manual/networks.md`)."""

    include: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        description="Catalog network ids or glob patterns (`fnmatch`, case-sensitive), e.g. "
        "`csiro_enwl/n01_*`. `<pattern>@<representation>` selects a named representation "
        "(`ieee8500@unbalanced`, `csiro_enwl/n01_f1@kron`) and overrides a plain match of the "
        "same network. Networks are ordered by id (seed tree).",
    )
    neutral_representation: Literal["explicit", "kron", "as_is"] = Field(
        default="explicit",
        description="`kron` loads the Kron-reduced representation where the source provides "
        "one (CSIRO); `explicit` and `as_is` load each network's default model.",
    )
    voltage_limits: VoltageLimits = VoltageLimits()


class LoadNormalisation(_Config):
    """Load model applied before extraction."""

    model: Literal["constant_power", "as_is"] = Field(
        default="constant_power",
        description="`constant_power`: every load becomes constant P, Q within "
        "[`vminpu`, `vmaxpu`]; `as_is`: the source's load models.",
    )
    vminpu: float = Field(
        default=0.6,
        gt=0,
        lt=1,
        description="Below this voltage (pu) OpenDSS turns a load into a constant impedance.",
    )
    vmaxpu: float = Field(
        default=1.5,
        gt=1,
        description="Above this voltage (pu) OpenDSS turns a load into a constant impedance.",
    )


class Normalisation(_Config):
    """Normalisation of the source models, recorded in the passports."""

    loads: LoadNormalisation = LoadNormalisation()
    s_base_va: SBase = Field(
        default="auto",
        description="Power base of the network (VA): per-unit base of the semantic model and the "
        "PMD export, scale of the positive-sequence view's source limits and balance tolerance "
        "(the view itself is per unit on 100 MVA). `auto`: the rating of the largest "
        "transformer of the source zone, else the total nominal load, else 1 MVA.",
    )


class Admittance(_Config):
    """Parameter perturbations of a variant; one draw per element."""

    line_length_scale: PositiveSpec = Field(
        default=Uniform(0.9, 1.1), description="Factor on each line's length."
    )
    neutral_impedance_scale: PositiveSpec = Field(
        default=Fixed(1.0),
        description="Factor on the neutral self and mutual impedances of explicit-neutral lines.",
    )
    grounding_resistance_ohm: PositiveSpec | None = Field(
        default=None,
        description="Resistance (Ω) of the existing grounding points; null keeps the source's.",
    )


class PhaseReallocation(_Config):
    """Moves single-phase customers to another phase available at their bus."""

    enabled: bool = Field(default=True, description="Reallocate phases in each variant.")
    fraction: float = Field(
        default=0.2, ge=0, le=1, description="Share of single-phase customers moved."
    )


class Variants(_Config):
    """Perturbed copies of each network; each must converge and pass the Kirchhoff residual
    check."""

    n_variants: int = Field(
        default=4,
        ge=1,
        le=MAX_VARIANTS,
        description="Variants per network (variant ids have four digits).",
    )
    admittance: Admittance = Admittance()
    phase_reallocation: PhaseReallocation = PhaseReallocation()


class PV(_Config):
    """Rooftop PV: placed per variant, output per scenario."""

    penetration: FractionSpec = Field(
        default=Uniform(0.0, 0.8),
        description="Share of customers with PV, drawn per variant. The default puts a "
        "voltage violation in about 6 % of the scenarios of a dataset of the whole catalog; "
        "0 places no PV.",
    )
    kwp: PositiveSpec = Field(
        default=Uniform(2.0, 6.0), description="Peak power (kW) of each PV system."
    )
    power_factor: PowerFactorSpec = Field(
        default=Fixed(1.0),
        description="Power factor of each inverter, drawn per variant; below 1 the inverter "
        "absorbs reactive power, within its rating.",
    )
    inverter_kva_per_kwp: float = Field(
        default=1.1, gt=0, description="Inverter rating (kVA) per kWp of PV."
    )
    reactive_fraction: FractionSpec = Field(
        default=Fixed(0.44),
        description="Largest reactive power of each inverter in the OPF, as a fraction of its "
        "rating (IEEE 1547-2018 Category B: 0.44; 1: the rating's circle only); drawn per "
        "variant and unit.",
    )
    reactive_threshold: FractionSpec = Field(
        default=Fixed(0.05),
        description="No reactive power in the OPF when a unit's available power is below this "
        "fraction of its rating (IEEE 1547-2018 Category B: 0.05; 0: always); drawn per variant "
        "and unit.",
    )


class DER(_Config):
    """Distributed energy resources added to the networks."""

    pv: PV = PV()


class SyntheticProfiles(_Config):
    """Synthetic loads: a global factor per scenario and lognormal noise per customer."""

    global_scale: NonNegativeSpec = Field(
        default=Uniform(0.2, 1.2), description="Factor on every nominal load, drawn per scenario."
    )
    local_sigma: float = Field(
        default=0.3, ge=0, description="Sigma of the lognormal noise on each customer's load."
    )
    power_factor: PowerFactorSpec = Field(
        default=Uniform(0.9, 0.99), description="Power factor of each load."
    )


class Profiles(_Config):
    """Where load values come from."""

    provider: Literal["synthetic", "network_loadshapes", "edp_smart_meter"] = Field(
        default="synthetic",
        description="`synthetic`; `network_loadshapes`: the loadshapes of the source model; "
        "`edp_smart_meter`: EDP's meter data.",
    )
    same_timestamp: bool = Field(
        default=True,
        description="All customers of a scenario share one timestamp (coincident demand).",
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
    """PV output profile."""

    provider: Literal["synthetic_clear_sky"] = Field(
        default="synthetic_clear_sky", description="Clear-sky profile reduced by cloudiness."
    )
    cloudiness: FractionSpec = Field(
        default=Uniform(0.0, 1.0), description="Cloudiness, drawn per scenario."
    )


class Scenarios(_Config):
    """Operating points of each variant."""

    n_scenarios_per_variant: int = Field(
        ge=1,
        le=MAX_SCENARIOS_PER_VARIANT,
        description="Scenarios per variant (scenario ids hold 31 bits of scenario index).",
    )
    profiles: Profiles = Profiles()
    pv_profile: PVProfile = PVProfile()
    source_voltage_pu: PositiveSpec = Field(
        default=Uniform(0.98, 1.05),
        description="Source voltage magnitude (pu), angle 0.",
    )


class Controls(_Config):
    """Regulator taps and capacitor states per scenario; labels always use
    `ControlMode=OFF`."""

    mode: Literal["static_settle", "as_is"] = Field(
        default="static_settle",
        description="`static_settle`: settle with `ControlMode=STATIC`, move the regulated "
        "taps by the tap jitter and settle again from there, freeze the state; `as_is`: the taps "
        "and capacitor states as compiled.",
    )
    tap_jitter_steps: int = Field(
        default=2,
        ge=0,
        description="Largest move (tap steps) of each settled tap before the controls settle "
        "again from there; 0 settles once.",
    )


class SampledCosts(_Config):
    """Cost coefficients of the preset ``sampled``, drawn per scenario (``c2 p^2 + c1 p``, ``p``
    in W): one pair for the source, one per PV unit."""

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
    """OPF labels: the minimum-cost OPF of each scenario (docs/manual/opf.md)."""

    enabled: bool = Field(
        default=False,
        description="Solve and verify the OPF of every scenario (needs the `opf` extra and "
        "constant-power loads).",
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
        description="Wall-clock limit of one OPF solve (s), Ipopt's `max_wall_time`.",
    )


class Labels(_Config):
    """What is solved for each scenario."""

    pf: bool = Field(default=True, description="Power-flow labels.")
    element_currents: bool = Field(
        default=False,
        description="Also store the current of every power-delivery element at each terminal "
        "conductor (`scenarios/element_results`); otherwise they are recomputed exactly as "
        "Yprim · V.",
    )
    opf: OPFLabels = OPFLabels()

    @model_validator(mode="after")
    def _something(self) -> Labels:
        if not self.pf and not self.opf.enabled:
            raise ValueError("nothing to label: enable pf or opf")
        return self


class Views(_Config):
    """Reduced views written next to the full model. The positive-sequence view is not
    one: ``export-posseq`` computes it from a dataset."""

    kron: bool = Field(
        default=False,
        description="Kron view (neutral eliminated), re-solved per scenario; ~1 kB per 100 phase "
        "nodes and scenario.",
    )


class Settings(_Config):
    """Execution settings; the dataset does not depend on the process and context counts."""

    data_dir: str = Field(
        default="./data",
        min_length=1,
        description="Directory of the datasets (relative to the working directory).",
    )
    seed: int = Field(ge=0, description="Root of the seed tree.")
    num_processes: int | None = Field(
        default=None,
        ge=1,
        description="Worker processes; null: one per core. With at least as many variants "
        "(networks x variants) as processes, a process runs all the chunks of a variant; with "
        "fewer, the chunks of a variant go to different processes.",
    )
    contexts_per_process: int = Field(
        default=1, ge=1, description="OpenDSS contexts (threads) per process."
    )
    chunk_size: int = Field(default=500, ge=1, description="Scenarios per output chunk.")
    float_precision: Literal["mixed", "full"] = Field(
        default="mixed",
        description="`mixed`: float32 for currents, powers and setpoints; `full`: float64 "
        "everywhere (docs/manual/outputs.md).",
    )
    residual_sample_rate: float = Field(
        default=0.01, ge=0, le=1, description="Share of scenarios with the full KCL residual check."
    )
    overwrite: bool = Field(
        default=False,
        description="Replace a dataset of the same name generated from another config.",
    )
    validation_tolerance: float = Field(
        default=1e-10,
        gt=0,
        le=MAX_TOLERANCE,
        description="OpenDSS `Tolerance` of the conformance runs; at most 1e-8 for the "
        "setpoint check.",
    )


class GenerationConfig(_Config):
    """A dataset generation config."""

    dataset_name: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Name of the dataset directory under `settings.data_dir`.",
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
