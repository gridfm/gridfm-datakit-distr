"""Semantic layer extraction: compiled circuit + electrical model -> :class:`SemanticModel`.

Series and shunt parameters come from the Yprims (exact for whatever the source used:
linecodes, geometries, sequence data); ratings, windings, taps and setpoints from the class
interfaces. Lines: ``Y_series = -Y_ft``, ``Z = inv(Y_series)``,
``Y_sh_fr = Y_ff + Y_ft``, ``Y_sh_to = Y_tt + Y_tf``, divided by the length in metres.

Mapping of the OpenDSS classes:

- Line -> ``line``; Line with ``Switch=yes`` -> ``switch`` (impedance in ohm);
- Transformer -> ``transformer`` (single-phase units stay separate, no bank merging);
- Capacitor, Reactor with all terminal-2 conductors grounded -> ``shunt`` (Yprim block of the
  non-ground conductors); a Reactor from neutral conductors to ground -> grounding of those bus
  terminals; series Capacitor or Reactor -> ``line`` with ``length_unit = "none"``;
- Load, Vsource, Generator, PVSystem (``solar``), Storage -> their PMD components.

Bus grounding: conductors tied to node 0 share one extra grounded terminal per bus (``rg = xg =
0``); a neutral conductor grounded through an impedance is a grounded terminal with that
impedance: ``Rneut + jXneut`` of a wye transformer winding, or the admittance of a grounding
reactor (its Yprim row sum). The electrical layer's ``r_ground_ohm`` (the row sum of ``Y_state``)
also carries couplings through transformers and anti-float terms.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from gridfm_datakit_distr.engine.context import DssContext
from gridfm_datakit_distr.extract.extract_electrical import GROUNDING_MIN_S
from gridfm_datakit_distr.model.electrical import GROUND, ElectricalModel, Element
from gridfm_datakit_distr.model.semantic import (
    Bus,
    ConnConfig,
    Generator,
    Line,
    Load,
    Matrix,
    RegControls,
    SemanticModel,
    Settings,
    Shunt,
    Solar,
    Storage,
    Switch,
    Transformer,
    VoltageSource,
)

TO_METRE = {1: 1609.344, 2: 304.8, 3: 1000.0, 4: 1.0, 5: 0.3048, 6: 0.0254, 7: 0.01, 8: 0.001}
"""OpenDSS length unit codes (mi, kft, km, m, ft, in, cm, mm; 0 = none) in metres."""
LOAD_MODELS = {1: "POWER", 2: "IMPEDANCE", 5: "CURRENT"}
LengthUnit = Literal["m", "none"]

ComplexMatrix = NDArray[np.complex128]


class SemanticError(ValueError):
    """The circuit cannot be described in the semantic layer."""


def _short(element: Element) -> str:
    return element.name.split(".", 1)[1]


def _real(m: ComplexMatrix) -> Matrix:
    return [[float(v) + 0.0 for v in row] for row in np.real(m)]  # + 0.0: no negative zeros


def _imag(m: ComplexMatrix) -> Matrix:
    return [[float(v) + 0.0 for v in row] for row in np.imag(m)]


def _per_phase(total: float, phases: int) -> list[float]:
    return [total / phases] * phases


class _Buses:
    """Terminal numbers of the buses and their grounding, as PMD's ENGINEERING model has them."""

    def __init__(self, model: ElectricalModel) -> None:
        self.model = model
        self.number = [int(n.opendss_node.rsplit(".", 1)[1]) for n in model.nodes]
        self.terminals: dict[str, set[int]] = defaultdict(set)
        for node, k in zip(model.nodes, self.number, strict=True):
            self.terminals[node.bus].add(k)
        # The extra terminal of conductors tied to node 0: 4 unless taken, as PMD's parser does.
        self.ground = {bus: max(*ks, 3) + 1 for bus, ks in self.terminals.items()}
        self.solid: set[str] = set()
        self.admittance: dict[int, complex] = defaultdict(complex)
        """Node index -> admittance to ground of grounding reactors and transformer neutrals (S)."""
        self.solid_nodes: set[int] = set()
        """Nodes grounded solidly through a transformer neutral (Rneut = Xneut = 0)."""

    def bus_of(self, element: Element, terminal: int) -> str:
        nodes = [k for k in element.terminals[terminal] if k != GROUND]
        if not nodes:
            raise SemanticError(f"{element.name}: terminal {terminal + 1} has only ground")
        return self.model.nodes[nodes[0]].bus

    def connections(self, element: Element, terminal: int, *, ground: bool = True) -> list[int]:
        """Terminal numbers of one element terminal; node 0 -> the bus's ground terminal (or
        dropped with ``ground=False``)."""
        bus = self.bus_of(element, terminal)
        out = []
        for k in element.terminals[terminal]:
            if k == GROUND:
                if ground:
                    out.append(self.ground[bus])
                    self.solid.add(bus)
                continue
            if self.model.nodes[k].bus != bus:
                raise SemanticError(f"{element.name}: terminal {terminal + 1} spans two buses")
            out.append(self.number[k])
        return out

    def add_reactor(self, element: Element) -> None:
        """A grounding reactor: its Yprim row sums are its admittance to ground."""
        assert element.yprim is not None
        local = element.local_nodes()
        keep = [p for p, k in enumerate(local) if k != GROUND]
        rows = element.yprim[:, keep].sum(axis=1)
        for p, k in enumerate(local):
            if k != GROUND:
                self.admittance[k] += complex(rows[p])

    def add_neutral(self, element: Element, winding: int, phases: int, z: complex) -> None:
        """The neutral of a wye winding grounded through ``z`` (Rneut + jXneut, ohm).

        Not the Yprim row sum: it also carries the coupling to other windings referred to ground
        (+0.3 % on the split-phase test circuit) and OpenDSS's anti-float admittance.
        """
        terminal = element.terminals[winding]
        if len(terminal) <= phases or terminal[phases] == GROUND:
            return  # neutral tied to node 0: a solid ground terminal already
        k = terminal[phases]
        if z == 0:
            self.solid_nodes.add(k)
        else:
            self.admittance[k] += 1.0 / z

    def build(self) -> dict[str, Bus]:
        grounded: dict[str, list[tuple[int, complex]]] = defaultdict(list)
        for k, y in self.admittance.items():
            if k not in self.solid_nodes and abs(y) >= GROUNDING_MIN_S:
                grounded[self.model.nodes[k].bus].append((self.number[k], 1.0 / y))
        for k in self.solid_nodes:
            grounded[self.model.nodes[k].bus].append((self.number[k], 0j))
        for bus in self.solid:
            grounded[bus].append((self.ground[bus], 0j))
        buses = {}
        for bus, numbers in self.terminals.items():
            g = sorted(grounded[bus])
            terminals = sorted(numbers | ({self.ground[bus]} if bus in self.solid else set()))
            buses[bus] = Bus(
                terminals=terminals,
                grounded=[t for t, _ in g],
                rg=[z.real + 0.0 for _, z in g],  # + 0.0: no negative zeros
                xg=[z.imag + 0.0 for _, z in g],
            )
        return buses


def is_grounding_reactor(element: Element, model: ElectricalModel) -> bool:
    """A reactor from neutral conductors to ground."""
    first = [k for k in element.terminals[0] if k != GROUND]
    rest = [k for t in element.terminals[1:] for k in t if k != GROUND]
    return bool(first) and not rest and all(model.nodes[k].conductor == "n" for k in first)


def _is_shunt(element: Element) -> bool:
    return all(k == GROUND for t in element.terminals[1:] for k in t)


def _series_blocks(element: Element) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix]:
    """(Z series, Y shunt at terminal 1, Y shunt at terminal 2) of a two-terminal element."""
    assert element.yprim is not None
    n = element.n_conductors
    y = element.yprim
    y_ft = y[:n, n:]
    try:
        z = np.asarray(np.linalg.inv(-y_ft), dtype=np.complex128)
    except np.linalg.LinAlgError as exc:
        raise SemanticError(f"{element.name}: singular series admittance") from exc
    return z, y[:n, :n] + y_ft, y[n:, n:] + y[n:, :n]


class _Extractor:
    def __init__(self, ctx: DssContext, model: ElectricalModel) -> None:
        self.ctx = ctx
        self.api = ctx.api
        self.model = model
        self.buses = _Buses(model)

    def prop(self, element: Element, name: str) -> str:
        """A property value by name (``Properties.Value``)."""
        self.ctx.activate_element(element.name)
        with self.ctx.engine_call(f"{element.name}.{name}"):
            return str(self.api.Properties.Value(name))

    def length(self, element: Element) -> tuple[float, LengthUnit]:
        """Length and its unit: metres if the line or its linecode declares a unit."""
        api = self.api
        with self.ctx.engine_call(f"Lines {element.name}"):
            api.Lines.Name(_short(element))
            length, unit = float(api.Lines.Length()), int(api.Lines.Units())
            code = str(api.Lines.LineCode())
            if unit == 0 and code:
                api.LineCodes.Name(code)
                unit = int(api.LineCodes.Units())
        if unit in TO_METRE:
            return length * TO_METRE[unit], "m"
        return length, "none"

    def _line_z(self, element: Element) -> ComplexMatrix:
        """Series impedance (ohm) from ``Lines.RMatrix/XMatrix`` x ``Lines.Length``."""
        api, n = self.api, element.n_conductors
        with self.ctx.engine_call(f"Lines.RMatrix {element.name}"):
            api.Lines.Name(_short(element))
            r = np.asarray(api.Lines.RMatrix(), dtype=np.float64).reshape(n, n)
            x = np.asarray(api.Lines.XMatrix(), dtype=np.float64).reshape(n, n)
            return np.asarray((r + 1j * x) * float(api.Lines.Length()), dtype=np.complex128)

    def line(self, element: Element, *, lumped: bool = False) -> Line:
        z, y_fr, y_to = _series_blocks(element)
        length: float = 1.0
        unit: LengthUnit = "none"
        if not lumped:
            length, unit = self.length(element)
        cm_ub = None if element.rating_a is None else [element.rating_a] * element.n_conductors
        return Line(
            source_id=element.name.lower(),
            f_bus=self.buses.bus_of(element, 0),
            t_bus=self.buses.bus_of(element, 1),
            f_connections=self.buses.connections(element, 0),
            t_connections=self.buses.connections(element, 1),
            rs=_real(z / length), xs=_imag(z / length),
            g_fr=_real(y_fr / length), b_fr=_imag(y_fr / length),
            g_to=_real(y_to / length), b_to=_imag(y_to / length),
            length=length, length_unit=unit, cm_ub=cm_ub,
        )  # fmt: skip

    def switch(self, element: Element) -> Switch:
        self.ctx.activate_element(element.name)
        with self.ctx.engine_call(f"IsOpen {element.name}"):
            ce = self.api.CktElement
            is_open = any(ce.IsOpen(t, 0) for t in (1, 2))
        # An open conductor's Yprim rows and columns are zero: take Z from the line data instead
        # (equal to inv(-Y_ft) of the closed switch to 1 ulp, confirmed by experiment).
        z = self._line_z(element) if is_open else _series_blocks(element)[0]
        cm_ub = None if element.rating_a is None else [element.rating_a] * element.n_conductors
        return Switch(
            source_id=element.name.lower(),
            f_bus=self.buses.bus_of(element, 0),
            t_bus=self.buses.bus_of(element, 1),
            f_connections=self.buses.connections(element, 0),
            t_connections=self.buses.connections(element, 1),
            rs=_real(z), xs=_imag(z), state="OPEN" if is_open else "CLOSED", cm_ub=cm_ub,
        )  # fmt: skip

    def shunt(self, element: Element, model: str) -> Shunt:
        assert element.yprim is not None
        keep = [p for p, k in enumerate(element.terminals[0]) if k != GROUND]
        block = element.yprim[np.ix_(keep, keep)]
        delta = self.prop(element, "conn").strip().lower().startswith("d")
        return Shunt(
            source_id=element.name.lower(),
            bus=self.buses.bus_of(element, 0),
            connections=self.buses.connections(element, 0, ground=False),
            configuration="DELTA" if delta else "WYE",
            gs=_real(block), bs=_imag(block),
            model=model,  # type: ignore[arg-type]
        )  # fmt: skip

    def regcontrols(self) -> dict[str, tuple[int, list[float]]]:
        """Regulated transformer (lower-case short name) -> (winding, settings)."""
        api = self.api
        out = {}
        with self.ctx.engine_call("RegControls"):
            i = api.RegControls.First()
            while i > 0:
                rc = api.RegControls
                getters = (rc.ForwardVreg, rc.ForwardBand, rc.PTRatio, rc.CTPrimary, rc.ForwardR,
                           rc.ForwardX)  # fmt: skip
                settings = [float(get()) for get in getters]
                out[str(api.RegControls.Transformer()).lower()] = (
                    int(api.RegControls.Winding()),
                    settings,
                )
                i = api.RegControls.Next()
        return out

    def transformer(
        self, element: Element, regulated: dict[str, tuple[int, list[float]]]
    ) -> Transformer:
        api = self.api
        self.ctx.activate_element(element.name)
        name = _short(element)
        with self.ctx.engine_call(f"Transformers {element.name}"):
            phases = int(api.CktElement.NumPhases())
            api.Transformers.Name(name)
            n = int(api.Transformers.NumWindings())
            if n > 3:
                raise SemanticError(f"{element.name}: {n} windings (at most 3 supported)")
            windings = []
            for w in range(1, n + 1):
                api.Transformers.Wdg(w)
                tr = api.Transformers
                windings.append(
                    (float(tr.kV()), float(tr.kVA()), float(tr.R()), float(tr.Tap()),
                     float(tr.MinTap()), float(tr.MaxTap()), int(tr.NumTaps()), bool(tr.IsDelta()))
                )  # fmt: skip
                if not tr.IsDelta() and float(tr.Rneut()) >= 0:  # Rneut < 0: open neutral
                    z_neutral = complex(float(tr.Rneut()), float(tr.Xneut()))
                    self.buses.add_neutral(element, w - 1, phases, z_neutral)
            xsc = [float(api.Transformers.Xhl())]
            if n == 3:
                xsc += [float(api.Transformers.Xht()), float(api.Transformers.Xlt())]
        sm_ub = float(self.prop(element, "emerghkva"))
        cmag = float(self.prop(element, "%imag")) / 100
        noloadloss = float(self.prop(element, "%noloadloss")) / 100
        leadlag = self.prop(element, "leadlag").strip().lower()
        reg = regulated.get(name.lower())
        controls = None
        if reg is not None:
            winding, settings = reg
            rows = [[[v] * phases if w == winding else [0.0] * phases for w in range(1, n + 1)]
                    for v in settings]  # fmt: skip
            controls = RegControls(vreg=rows[0], band=rows[1], ptratio=rows[2], ctprim=rows[3],
                                   r=rows[4], x=rows[5])  # fmt: skip
        configuration: list[ConnConfig] = ["DELTA" if w[7] else "WYE" for w in windings]
        return Transformer(
            source_id=element.name.lower(),
            bus=[self.buses.bus_of(element, t) for t in range(n)],
            # A delta winding's fourth conductor (OpenDSS ties it to node 0) carries nothing.
            connections=[
                self.buses.connections(element, t, ground=configuration[t] == "WYE")
                for t in range(n)
            ],
            configuration=configuration,
            polarity=[1] * n,
            leadlag="lead" if leadlag.startswith("lead") else "lag",
            vm_nom=[w[0] for w in windings],
            sm_nom=[w[1] for w in windings],
            sm_ub=sm_ub,
            rw=[w[2] / 100 for w in windings],
            xsc=[x / 100 for x in xsc],
            cmag=cmag,
            noloadloss=noloadloss,
            tm_set=[[w[3]] * phases for w in windings],
            tm_lb=[[w[4]] * phases for w in windings],
            tm_ub=[[w[5]] * phases for w in windings],
            tm_step=[[(w[5] - w[4]) / w[6] if w[6] else 0.0] * phases for w in windings],
            tm_fix=[[reg is None or reg[0] != i] * phases for i in range(1, n + 1)],
            controls=controls,
        )

    def load(self, element: Element) -> Load:
        api = self.api
        with self.ctx.engine_call(f"Loads {element.name}"):
            api.Loads.Name(_short(element))
            model_number = int(api.Loads.Model())
            phases = int(api.Loads.Phases())
            kw, kvar, kv = float(api.Loads.kW()), float(api.Loads.kvar()), float(api.Loads.kV())
            delta = bool(api.Loads.IsDelta())
        if model_number not in LOAD_MODELS:
            raise SemanticError(f"{element.name}: load model {model_number} (normalise loads)")
        return Load(
            source_id=element.name.lower(),
            bus=self.buses.bus_of(element, 0),
            connections=self.buses.connections(element, 0, ground=not delta),
            configuration="DELTA" if delta else "WYE",
            model=LOAD_MODELS[model_number],  # type: ignore[arg-type]
            pd_nom=_per_phase(kw, phases),
            qd_nom=_per_phase(kvar, phases),
            vm_nom=kv,
        )

    def voltage_source(self, element: Element) -> tuple[VoltageSource, float]:
        """The source and its line-to-neutral voltage base (kV)."""
        api = self.api
        with self.ctx.engine_call(f"Vsources {element.name}"):
            api.Vsources.Name(_short(element))
            base_kv, pu = float(api.Vsources.BasekV()), float(api.Vsources.PU())
            angle, phases = float(api.Vsources.AngleDeg()), int(api.Vsources.Phases())
        if not _is_shunt(element):
            raise SemanticError(f"{element.name}: bus2 is not ground")
        v_ln = base_kv / math.sqrt(3) if phases > 1 else base_kv
        y = self.ctx.element_yprim(element.name)
        n = element.n_conductors
        z = np.zeros((n + 1, n + 1), dtype=complex)
        z[:n, :n] = np.linalg.inv(y[:n, :n])
        phase_terminals = self.buses.connections(element, 0)
        bus = self.buses.bus_of(element, 0)
        self.buses.solid.add(bus)
        source = VoltageSource(
            source_id=element.name.lower(),
            bus=bus,
            connections=[*phase_terminals, self.buses.ground[bus]],
            configuration="WYE",
            vm=[v_ln * pu] * n + [0.0],
            va=[(angle - 120.0 * k + 180.0) % 360.0 - 180.0 for k in range(n)] + [0.0],
            rs=_real(z), xs=_imag(z),
        )  # fmt: skip
        return source, v_ln

    def injection_common(self, element: Element) -> tuple[str, list[int], ConnConfig, int]:
        delta = self.prop(element, "conn").strip().lower().startswith("d")
        self.ctx.activate_element(element.name)
        with self.ctx.engine_call(f"NumPhases {element.name}"):
            phases = int(self.api.CktElement.NumPhases())
        connections = self.buses.connections(element, 0, ground=not delta)
        return self.buses.bus_of(element, 0), connections, "DELTA" if delta else "WYE", phases

    def generator(self, element: Element) -> Generator:
        bus, connections, configuration, phases = self.injection_common(element)
        api = self.api
        with self.ctx.engine_call(f"Generators {element.name}"):
            api.Generators.Name(_short(element))
            kw, kvar = float(api.Generators.kW()), float(api.Generators.kvar())
            kva = float(api.Generators.kVARated())
        return Generator(source_id=element.name.lower(), bus=bus, connections=connections,
                         configuration=configuration, pg=_per_phase(kw, phases),
                         qg=_per_phase(kvar, phases), sm_ub=kva)  # fmt: skip

    def solar(self, element: Element) -> Solar:
        bus, connections, configuration, phases = self.injection_common(element)
        api = self.api
        with self.ctx.engine_call(f"PVsystems {element.name}"):
            api.PVsystems.Name(_short(element))
            kw, kvar = float(api.PVsystems.kW()), float(api.PVsystems.kvar())
            available = float(api.PVsystems.Pmpp()) * float(api.PVsystems.Irradiance())
            kva = float(api.PVsystems.kVARated())
        return Solar(source_id=element.name.lower(), bus=bus, connections=connections,
                     configuration=configuration, pg=_per_phase(kw, phases),
                     qg=_per_phase(kvar, phases), pg_ub=_per_phase(available, phases),
                     sm_ub=kva)  # fmt: skip

    def storage(self, element: Element) -> Storage:
        bus, connections, configuration, _ = self.injection_common(element)
        kwh = float(self.prop(element, "kWhrated"))
        kw = float(self.prop(element, "kWrated"))
        stored = float(self.prop(element, "%stored")) / 100
        kva = float(self.prop(element, "kva"))
        return Storage(source_id=element.name.lower(), bus=bus, connections=connections,
                       configuration=configuration, energy=kwh * stored, energy_ub=kwh,
                       charge_ub=kw, discharge_ub=kw, sm_ub=kva)  # fmt: skip


def extract_semantic(ctx: DssContext, model: ElectricalModel, *, s_base_va: float) -> SemanticModel:
    """Describe the compiled circuit of ``ctx`` (from which ``model`` was extracted)."""
    x = _Extractor(ctx, model)
    regulated = x.regcontrols()
    parts: dict[str, dict[str, object]] = defaultdict(dict)
    vbases: dict[str, float] = {}

    def put(kind: str, name: str, value: object) -> None:
        if name in parts[kind]:
            raise SemanticError(f"two elements named {name!r} in {kind}")
        parts[kind][name] = value

    makers: dict[str, Callable[[Element], tuple[str, object]]] = {
        "line": lambda e: ("line", x.line(e)),
        "switch": lambda e: ("switch", x.switch(e)),
        "load": lambda e: ("load", x.load(e)),
        "generator": lambda e: ("generator", x.generator(e)),
        "pvsystem": lambda e: ("solar", x.solar(e)),
        "storage": lambda e: ("storage", x.storage(e)),
    }
    for e in model.elements:
        cls = e.element_class
        name = _short(e).lower()
        if cls in makers:
            kind, value = makers[cls](e)
            put(kind, name, value)
        elif cls == "transformer":
            put("transformer", name, x.transformer(e, regulated))
        elif cls in ("capacitor", "reactor"):
            if cls == "reactor" and is_grounding_reactor(e, model):
                x.buses.add_reactor(e)
            elif _is_shunt(e):
                put("shunt", name, x.shunt(e, "CAPACITOR" if cls == "capacitor" else "REACTOR"))
            else:
                put("line", name, x.line(e, lumped=True))
        elif cls == "vsource":
            source, v_ln = x.voltage_source(e)
            put("voltage_source", name, source)
            vbases[source.bus] = v_ln
        else:
            raise SemanticError(f"{e.name}: class {cls} has no semantic mapping")
    with ctx.engine_call("Solution.Frequency"):
        frequency = float(ctx.api.Solution.Frequency())
    settings = Settings(sbase_default=s_base_va / 1e3, vbases_default=vbases,
                        base_frequency=frequency)  # fmt: skip
    return SemanticModel(name=model.variant_id, settings=settings, bus=x.buses.build(),
                         **parts)  # type: ignore[arg-type]  # fmt: skip
