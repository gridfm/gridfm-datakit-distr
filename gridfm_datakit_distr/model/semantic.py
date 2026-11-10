"""Semantic layer: a PMD ENGINEERING-aligned description of one network variant.

Field names and units follow the ENGINEERING model of PowerModelsDistribution 0.16 as its own
parser fills it: voltages in kV and powers in kW, kvar,
kVA (``settings.voltage_scale_factor`` = ``power_scale_factor`` = 1000), impedances in ohm, line
parameters per metre with ``length`` in metres, angles in degrees, enums as upper-case strings.
Matrices are lists of rows. Terminals are OpenDSS node numbers; conductors that OpenDSS ties to
node 0 go to one extra terminal of their bus listed in ``grounded`` with ``rg = xg = 0``, as
PMD's parser does.

Extensions (not PMD fields):

- ``line.length_unit``: ``"m"``, or ``"none"`` when neither the line nor its linecode declares a
  unit (IEEE 37, lumped series elements): ``length`` and the per-length values are then in the
  source's own unit and only their product is physical;
- ``transformer.leadlag``: the OpenDSS phase-shift convention of delta-wye windings; PMD encodes
  it with ``polarity`` and the order of ``connections``, which the exporter sets.

Generator, solar and storage fields are placeholders until the DER specification and the
PMD exporter; none of the catalog networks has such elements.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Matrix = list[list[float]]
Status = Literal["ENABLED", "DISABLED"]
ConnConfig = Literal["WYE", "DELTA"]
Dispatchable = Literal["YES", "NO"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_square(name: str, matrix: Matrix, n: int) -> None:
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"{name} must be {n}x{n}")


class Bus(_Strict):
    terminals: list[int]
    grounded: list[int] = Field(default_factory=list)
    rg: list[float] = Field(default_factory=list)
    """Grounding resistance of each grounded terminal (ohm)."""
    xg: list[float] = Field(default_factory=list)
    status: Status = "ENABLED"

    @model_validator(mode="after")
    def _grounding(self) -> Bus:
        if not set(self.grounded) <= set(self.terminals):
            raise ValueError("grounded terminals must be terminals of the bus")
        if not len(self.rg) == len(self.xg) == len(self.grounded):
            raise ValueError("rg and xg need one value per grounded terminal")
        return self


class _Element(_Strict):
    source_id: str
    """OpenDSS element name, lower case (``line.650632``)."""
    status: Status = "ENABLED"


class _Branch(_Element):
    f_bus: str
    t_bus: str
    f_connections: list[int]
    t_connections: list[int]
    rs: Matrix
    xs: Matrix
    cm_ub: list[float] | None = None
    """Normal ampacity of each conductor (A)."""

    @model_validator(mode="after")
    def _sizes(self) -> _Branch:
        n = len(self.f_connections)
        if len(self.t_connections) != n:
            raise ValueError("f_connections and t_connections differ in length")
        _check_square("rs", self.rs, n)
        _check_square("xs", self.xs, n)
        if self.cm_ub is not None and len(self.cm_ub) != n:
            raise ValueError("cm_ub needs one value per conductor")
        return self


class Line(_Branch):
    """Series impedance (ohm per length unit) and shunt admittances at each end (S per unit)."""

    g_fr: Matrix
    b_fr: Matrix
    g_to: Matrix
    b_to: Matrix
    length: float = Field(gt=0)
    length_unit: Literal["m", "none"]

    @model_validator(mode="after")
    def _shunt_sizes(self) -> Line:
        n = len(self.f_connections)
        for name in ("g_fr", "b_fr", "g_to", "b_to"):
            _check_square(name, getattr(self, name), n)
        return self


class Switch(_Branch):
    """Series impedance in ohm (no length)."""

    state: Literal["CLOSED", "OPEN"] = "CLOSED"
    dispatchable: Dispatchable = "NO"


class RegControls(_Strict):
    """RegControl settings per winding and phase; zero on windings that are not regulated."""

    vreg: Matrix
    band: Matrix
    ptratio: Matrix
    ctprim: Matrix
    r: Matrix
    x: Matrix


class Transformer(_Element):
    bus: list[str]
    connections: list[list[int]]
    configuration: list[ConnConfig]
    polarity: list[int]
    leadlag: Literal["lag", "lead"]
    vm_nom: list[float]
    """Winding voltage (kV): line-to-line for three-phase units, winding voltage for one phase."""
    sm_nom: list[float]
    """Winding rating (kVA)."""
    sm_ub: float
    """Emergency rating of the first winding (kVA)."""
    rw: list[float]
    """Winding resistances (pu of the winding rating)."""
    xsc: list[float]
    """Short-circuit reactances between winding pairs (1-2, 1-3, ..., 2-3, ...), pu of winding 1."""
    cmag: float
    noloadloss: float
    tm_set: Matrix
    tm_lb: Matrix
    tm_ub: Matrix
    tm_step: Matrix
    tm_fix: list[list[bool]]
    controls: RegControls | None = None

    @model_validator(mode="after")
    def _windings(self) -> Transformer:
        n = len(self.bus)
        per_winding = (self.connections, self.configuration, self.polarity, self.vm_nom,
                       self.sm_nom, self.rw, self.tm_set, self.tm_lb, self.tm_ub, self.tm_step,
                       self.tm_fix)  # fmt: skip
        if n < 2 or any(len(v) != n for v in per_winding):
            raise ValueError("every per-winding field needs one entry per winding (>= 2)")
        if len(self.xsc) != n * (n - 1) // 2:
            raise ValueError("xsc needs one value per winding pair")
        return self


class Shunt(_Element):
    """Admittance matrix (S) between the connections and ground."""

    bus: str
    connections: list[int]
    configuration: ConnConfig
    gs: Matrix
    bs: Matrix
    model: Literal["CAPACITOR", "REACTOR", "GENERIC"]
    dispatchable: Dispatchable = "NO"

    @model_validator(mode="after")
    def _sizes(self) -> Shunt:
        _check_square("gs", self.gs, len(self.connections))
        _check_square("bs", self.bs, len(self.connections))
        return self


class _Injection(_Element):
    bus: str
    connections: list[int]
    configuration: ConnConfig


class Load(_Injection):
    model: Literal["POWER", "IMPEDANCE", "CURRENT"]
    pd_nom: list[float]
    """Active power per phase (kW)."""
    qd_nom: list[float]
    vm_nom: float
    """Rated voltage (kV) as in OpenDSS: line-to-line unless single-phase wye."""
    dispatchable: Dispatchable = "NO"

    @model_validator(mode="after")
    def _phases(self) -> Load:
        if not self.pd_nom or len(self.pd_nom) != len(self.qd_nom):
            raise ValueError("pd_nom and qd_nom need one value per phase")
        return self


class VoltageSource(_Injection):
    vm: list[float]
    """Voltage magnitude per connection (kV); 0 on the ground terminal."""
    va: list[float]
    """Angle per connection (degrees)."""
    rs: Matrix
    xs: Matrix

    @model_validator(mode="after")
    def _sizes(self) -> VoltageSource:
        n = len(self.connections)
        if len(self.vm) != n or len(self.va) != n:
            raise ValueError("vm and va need one value per connection")
        _check_square("rs", self.rs, n)
        _check_square("xs", self.xs, n)
        return self


class Generator(_Injection):
    pg: list[float]
    """Active power setpoint per phase (kW, delivered)."""
    qg: list[float]
    sm_ub: float
    """Rating (kVA)."""


class Solar(_Injection):
    pg: list[float]
    """Active power output per phase (kW)."""
    qg: list[float]
    pg_ub: list[float]
    """Available power per phase (kW): Pmpp times irradiance."""
    sm_ub: float
    """Inverter rating (kVA)."""


class Storage(_Injection):
    energy: float
    """Stored energy (kWh)."""
    energy_ub: float
    charge_ub: float
    """Rated power (kW)."""
    discharge_ub: float
    sm_ub: float


class Settings(_Strict):
    sbase_default: float = Field(gt=0)
    """Power base (kVA): ``s_base_va`` / 1000."""
    vbases_default: dict[str, float]
    """Line-to-neutral voltage base of the source bus (kV)."""
    voltage_scale_factor: float = 1000.0
    power_scale_factor: float = 1000.0
    base_frequency: float


class SemanticModel(_Strict):
    data_model: Literal["ENGINEERING"] = "ENGINEERING"
    name: str
    """Variant id."""
    settings: Settings
    bus: dict[str, Bus]
    line: dict[str, Line] = Field(default_factory=dict)
    switch: dict[str, Switch] = Field(default_factory=dict)
    transformer: dict[str, Transformer] = Field(default_factory=dict)
    shunt: dict[str, Shunt] = Field(default_factory=dict)
    load: dict[str, Load] = Field(default_factory=dict)
    voltage_source: dict[str, VoltageSource] = Field(default_factory=dict)
    generator: dict[str, Generator] = Field(default_factory=dict)
    solar: dict[str, Solar] = Field(default_factory=dict)
    storage: dict[str, Storage] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _references(self) -> SemanticModel:
        """Every element connects to existing buses through existing terminals."""

        def check(where: str, bus: str, connections: list[int]) -> None:
            if bus not in self.bus:
                raise ValueError(f"{where}: unknown bus {bus!r}")
            missing = set(connections) - set(self.bus[bus].terminals)
            if missing:
                raise ValueError(f"{where}: bus {bus!r} has no terminals {sorted(missing)}")

        for kind, branches in (("line", self.line), ("switch", self.switch)):
            for name, b in branches.items():
                check(f"{kind}.{name}", b.f_bus, b.f_connections)
                check(f"{kind}.{name}", b.t_bus, b.t_connections)
        for name, t in self.transformer.items():
            for bus, connections in zip(t.bus, t.connections, strict=True):
                check(f"transformer.{name}", bus, connections)
        injections: dict[str, dict[str, _Injection]] = {
            "load": dict(self.load), "voltage_source": dict(self.voltage_source),
            "generator": dict(self.generator), "solar": dict(self.solar),
            "storage": dict(self.storage),
        }  # fmt: skip
        for kind, items in injections.items():
            for name, item in items.items():
                check(f"{kind}.{name}", item.bus, item.connections)
        for name, s in self.shunt.items():
            check(f"shunt.{name}", s.bus, s.connections)
        if set(self.settings.vbases_default) - set(self.bus):
            raise ValueError("settings.vbases_default names an unknown bus")
        return self
