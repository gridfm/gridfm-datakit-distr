"""Tiny OpenDSS circuits written inline for the tests (our own data)."""

from __future__ import annotations

from pathlib import Path


def write_circuit(directory: Path, files: dict[str, str], master: str = "master.dss") -> Path:
    """Write ``files`` (relative path -> text) under ``directory``; return the master path."""
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return directory / master


# Balanced 0.4 kV source -> 1 km line -> wye load with constant impedance (model=2).
# The circuit is linear and balanced, so per phase V_load / V_src = Z_load / (Z_line + Z_load)
# with Z_line = 0.1 + j0.08 ohm and Z_load = (400 V)^2 / conj(10 kW + j5 kvar).
# The line lives in a sub-folder to exercise relative includes.
TWO_BUS_Z_LINE_OHM = 0.1 + 0.08j
TWO_BUS_Z_LOAD_OHM = 400.0**2 / (10e3 - 5e3j)  # V_ll^2 / conj(S_3ph)
TWO_BUS = {
    "master.dss": """\
Clear
New Circuit.two_bus basekv=0.4 pu=1.0 phases=3 bus1=src angle=0
Redirect lines/line.dss
New Load.ld bus1=load.1.2.3 phases=3 kV=0.4 kW=10 kvar=5 model=2 conn=wye
Set VoltageBases=[0.4]
CalcVoltageBases
""",
    "lines/line.dss": """\
New Line.l1 bus1=src.1.2.3 bus2=load.1.2.3 phases=3 r1=0.1 x1=0.08 r0=0.3 x0=0.2 c1=0 c0=0
~ length=1 units=km
""",
}

# 11 kV feeder with a single-phase lateral and constant-power loads (nonlinear PF).
LATERAL = {
    "master.dss": """\
Clear
New Circuit.lateral basekv=11 pu=1.02 phases=3 bus1=sub angle=0
New Line.main bus1=sub.1.2.3 bus2=n1.1.2.3 phases=3 r1=0.3 x1=0.4 r0=0.6 x0=1.2 c1=0 c0=0
~ length=2 units=km
New Line.lat bus1=n1.2 bus2=n2.2 phases=1 r1=0.5 x1=0.3 r0=0.5 x0=0.3 c1=0 c0=0
~ length=1 units=km
New Load.l3 bus1=n1.1.2.3 phases=3 kV=11 kW=500 kvar=200 model=1
New Load.l1 bus1=n2.2 phases=1 kV=6.351 kW=100 kvar=30 model=1
Set VoltageBases=[11]
CalcVoltageBases
""",
}


# 0.4 kV feeder loaded until its far end sits well below 0.95 pu, where constant-power loads
# leave their model at the default vminpu: models 1, 2 (constant Z) and 5 (constant I),
# a delta load, a single-phase load and a disabled load.
HEAVY = {
    "master.dss": """\
Clear
New Circuit.heavy basekv=0.4 pu=1.0 phases=3 bus1=src angle=0 MVAsc3=100 MVAsc1=100
New Line.l1 bus1=src bus2=a phases=3 r1=0.1 x1=0.05 r0=0.3 x0=0.15 c1=0 c0=0 length=0.4 units=km
New Line.l2 bus1=a bus2=b phases=3 r1=0.1 x1=0.05 r0=0.3 x0=0.15 c1=0 c0=0 length=0.4 units=km
New Load.a3 bus1=a phases=3 kV=0.4 kW=60 kvar=20 model=1
New Load.b1 bus1=b.1 phases=1 kV=0.23 kW=20 kvar=6 model=1
New Load.bd bus1=b phases=3 kV=0.4 kW=30 kvar=10 model=1 conn=delta
New Load.bz bus1=b phases=3 kV=0.4 kW=20 kvar=5 model=2
New Load.bi bus1=b phases=3 kV=0.4 kW=20 kvar=5 model=5
New Load.off bus1=b phases=3 kV=0.4 kW=5 kvar=1 model=2
Disable Load.off
Set VoltageBases=[0.4]
CalcVoltageBases
""",
}


# 0.4 kV four-wire feeder: the neutral src.4 is grounded through 0.5 + j0.2 ohm only at the
# source; a single-phase phase-to-neutral load and a three-phase delta load.
FOUR_WIRE_GROUND_OHM = 0.5 + 0.2j
FOUR_WIRE = {
    "master.dss": """\
Clear
New Circuit.four_wire basekv=0.4 pu=1.0 phases=3 bus1=src.1.2.3 angle=0 MVAsc3=100 MVAsc1=100
New Reactor.ground phases=1 bus1=src.4 bus2=src.0 R=0.5 X=0.2
New Linecode.lc4 nphases=4 units=km
~ rmatrix=(0.5 | 0.1 0.5 | 0.1 0.1 0.5 | 0.1 0.1 0.1 0.6)
~ xmatrix=(0.3 | 0.1 0.3 | 0.1 0.1 0.3 | 0.1 0.1 0.1 0.35)
~ cmatrix=(0 | 0 0 | 0 0 0 | 0 0 0 0)
New Line.l1 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 phases=4 linecode=lc4 length=0.2 units=km
New Load.house bus1=b1.1.4 phases=1 kV=0.23 kW=3 kvar=1 model=1
New Load.delta bus1=b1.1.2.3 phases=3 conn=delta kV=0.4 kW=6 kvar=2 model=1
Set VoltageBases=[0.4]
CalcVoltageBases
""",
}

# Single-phase 12.47 kV lateral with a regulator (RegControl), a switch, a disabled line, a
# capacitor with CapControl, a load and an energy meter.
CONTROLLED = {
    "master.dss": """\
Clear
New Circuit.ctrl basekv=12.47 pu=1.0 phases=3 bus1=src
New Transformer.reg phases=1 windings=2 buses=[src.1 r.1] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[1000 1000] XHL=0.1 %LoadLoss=0.01
New RegControl.reg transformer=reg winding=2 vreg=122 band=2 ptratio=60
New Line.l1 bus1=r.1 bus2=a.1 phases=1 r1=0.3 x1=0.6 r0=0.3 x0=0.6 c1=0 c0=0 length=1 units=km
New Line.sw bus1=a.1 bus2=b.1 phases=1 switch=yes
New Line.off bus1=a.1 bus2=b.1 phases=1 r1=1 x1=1 r0=1 x0=1 c1=0 c0=0 length=1 units=km
Disable Line.off
New Capacitor.cap bus1=b.1 phases=1 kvar=100 kv=7.2
New CapControl.cc capacitor=cap element=Line.l1 terminal=1 type=voltage ON=118 OFF=125 PTratio=60
New Load.ld bus1=b.1 phases=1 kV=7.2 kW=200 kvar=50 model=1
New EnergyMeter.m element=Line.l1 terminal=1
Set VoltageBases=[12.47]
CalcVoltageBases
""",
}


# Adversarial layout (pattern from gridfm-datakit's ordering tests): elements defined far end
# first, buses with missing phases (m2 has only .2 .3, far only .3), a distinct load on every
# phase and a phase-to-phase load. A wrong node mapping breaks the residual and the C3 check.
ADVERSARIAL = {
    "master.dss": """\
Clear
New Circuit.adv basekv=11 pu=1.0 phases=3 bus1=src angle=0
New Line.far bus1=m2.3 bus2=far.3 phases=1 r1=0.4 x1=0.3 r0=0.4 x0=0.3 c1=0 c0=0 length=1 units=km
New Line.m1m2 bus1=m1.2.3 bus2=m2.2.3 phases=2 r1=0.3 x1=0.2 r0=0.5 x0=0.4 c1=0 c0=0
~ length=1 units=km
New Line.trunk bus1=src.1.2.3 bus2=m1.1.2.3 phases=3 r1=0.2 x1=0.3 r0=0.4 x0=0.9 c1=0 c0=0
~ length=2 units=km
New Load.a bus1=m1.1 phases=1 kV=6.351 kW=101 kvar=11 model=1
New Load.b bus1=m2.2 phases=1 kV=6.351 kW=203 kvar=23 model=1
New Load.c bus1=far.3 phases=1 kV=6.351 kW=307 kvar=37 model=1
New Load.bc bus1=m2.2.3 phases=1 kV=11 kW=59 kvar=7 model=1
Set VoltageBases=[11]
CalcVoltageBases
""",
}


# Every supported class in one well-posed circuit: a geometry line (Kron-reduced), a matrix
# line, a 3-winding transformer with a delta tertiary, a centre-tap transformer (split phase), a
# bank of single-phase transformers, series and shunt reactors, a delta capacitor, wye, delta and
# split-phase loads, a generator, a PV system and a storage unit. The generator sits on a grounded
# wye bus: a constant-power wye generator on the ungrounded delta tertiary needs ~350 iterations.
ALL_CLASSES = {
    "master.dss": """\
Clear
New Circuit.cls basekv=12.47 pu=1.0 phases=3 bus1=src angle=0
New Wiredata.acsr1 GMR=0.0244 DIAM=0.721 RAC=0.306 NormAmps=530 Runits=mi radunits=in gmrunits=ft
New Linegeometry.g4 nconds=4 nphases=3 reduce=y
~ cond=1 wire=acsr1 x=-4 h=28 units=ft
~ cond=2 wire=acsr1 x=-1.5 h=28 units=ft
~ cond=3 wire=acsr1 x=3 h=28 units=ft
~ cond=4 wire=acsr1 x=0 h=24 units=ft
New Line.geo bus1=src.1.2.3 bus2=g.1.2.3 geometry=g4 length=1 units=kft
New Line.mat bus1=g.1.2.3 bus2=m.1.2.3 phases=3 rmatrix=(0.3 | 0.1 0.3 | 0.1 0.1 0.3)
~ xmatrix=(0.8 | 0.3 0.8 | 0.3 0.3 0.8) cmatrix=(10 | -2 10 | -2 -2 10) length=1 units=km
New Transformer.t3w phases=3 windings=3 buses=[m t3a t3b] conns=[wye wye delta]
~ kvs=[12.47 4.16 0.48] kvas=[3000 2000 1000] xhl=7 xht=8 xlt=3 %loadloss=1
New Transformer.ct phases=1 windings=3 buses=[m.1 ct.1.0 ct.0.2] conns=[wye wye wye]
~ kvs=[7.2 0.12 0.12] kvas=[50 50 50] %r=0.6 xhl=2.04 xht=2.04 xlt=1.36
New Transformer.ba phases=1 windings=2 buses=[m.1 bk.1] kvs=[7.2 2.4] kvas=[500 500] xhl=2 bank=bk
New Transformer.bb phases=1 windings=2 buses=[m.2 bk.2] kvs=[7.2 2.4] kvas=[500 500] xhl=2 bank=bk
New Transformer.bc phases=1 windings=2 buses=[m.3 bk.3] kvs=[7.2 2.4] kvas=[500 500] xhl=2 bank=bk
New Reactor.series bus1=t3a.1.2.3 bus2=sr.1.2.3 phases=3 R=0.5 X=2
New Reactor.shunt bus1=sr.1.2.3 phases=3 kvar=100 kv=4.16
New Capacitor.cdelta bus1=t3b.1.2.3 phases=3 kvar=60 kv=0.48 conn=delta
New Load.l3 bus1=sr.1.2.3 phases=3 kV=4.16 kW=300 kvar=100 model=1
New Load.ldelta bus1=t3b.1.2.3 phases=3 kV=0.48 kW=50 kvar=10 model=1 conn=delta
New Load.lct bus1=ct.1.2 phases=1 kV=0.24 kW=5 kvar=1 model=1
New Load.lbk bus1=bk.1.2.3 phases=3 kV=4.16 kW=400 kvar=100 model=1
New Generator.g bus1=sr.1.2.3 phases=3 kV=4.16 kW=100 kvar=20 model=1
New PVSystem.pv bus1=bk.1.2.3 phases=3 kV=4.16 kVA=120 Pmpp=100 irradiance=0.8 pf=1
New Storage.st bus1=sr.1.2.3 phases=3 kV=4.16 kWrated=50 kWhrated=200 %stored=50
~ State=DISCHARGING %Discharge=50
Set VoltageBases=[12.47 4.16 0.48 0.24]
CalcVoltageBases
""",
}


_LV_FEEDER = """\
New Linecode.lc4 nphases=4 units=km
~ rmatrix=(0.5 | 0.1 0.5 | 0.1 0.1 0.5 | 0.1 0.1 0.1 0.6)
~ xmatrix=(0.3 | 0.1 0.3 | 0.1 0.1 0.3 | 0.1 0.1 0.1 0.35)
~ cmatrix=(0 | 0 0 | 0 0 0 | 0 0 0 0)
New Line.lv bus1=lv.1.2.3.4 bus2=c.1.2.3.4 phases=4 linecode=lc4 length=0.1 units=km
New Load.house bus1=c.1.4 phases=1 kV=0.23 kW=8 kvar=2 model=1
"""


def mv_lv(transformer_extra: str, mv_load: str = "") -> dict[str, str]:
    """22/0.4 kV delta-wye transformer feeding a four-wire LV line (EDP-like)."""
    return {
        "master.dss": f"""\
Clear
New Circuit.mvlv basekv=22 pu=1.0 phases=3 bus1=mv
New Line.mv bus1=mv.1.2.3 bus2=mv2.1.2.3 phases=3 r1=0.3 x1=0.3 r0=0.6 x0=1 c1=0 c0=0
~ length=1 units=km
{mv_load}
New Transformer.t windings=2 Buses=[mv2.1.2.3 lv.1.2.3.4] Conns=[Delta Wye] kVs=[22 0.4]
~ kVAs=[250 250] XHL=4 %Rs=[0.05 0.05] {transformer_extra}
{_LV_FEEDER}
Set VoltageBases=[22 0.4]
CalcVoltageBases
""",
    }


# Transformer neutral grounded through Rneut + jXneut (EDP rural style): inside the Yprim.
RNEUT_OHM = 5 + 0.01j
MV_LV_RNEUT = mv_lv("Rneut=5 Xneut=0.01")
# Rneut left at its default (-1, open): the LV neutral floats (EDP industrial style).
MV_LV_FLOATING = mv_lv("")
# A wye load grounded on the MV side (Kron pattern) plus an explicit LV neutral: mixed.
MV_LV_MIXED = mv_lv(
    "Rneut=5 Xneut=0.01", "New Load.mvload bus1=mv2.1.2.3 phases=3 kV=22 kW=100 kvar=30 model=1"
)


# North-American split-phase service with an explicit centre node (x.3) grounded at the
# transformer, a three-wire drop (s1, s2, n) and 120 V / 240 V loads.
SPLIT_PHASE = {
    "master.dss": """\
Clear
New Circuit.split basekv=12.47 pu=1.0 phases=3 bus1=src
New Line.mv bus1=src.1 bus2=p.1 phases=1 r1=0.3 x1=0.6 r0=0.3 x0=0.6 c1=0 c0=0 length=1 units=km
New Transformer.ct phases=1 windings=3 buses=[p.1 x.1.3 x.3.2] conns=[wye wye wye]
~ kvs=[7.2 0.12 0.12] kvas=[25 25 25] %r=0.6 xhl=2.04 xht=2.04 xlt=1.36
New Reactor.gnd phases=1 bus1=x.3 bus2=x.0 R=0.01 X=0
New Linecode.tpx nphases=3 units=kft rmatrix=(0.6 | 0.1 0.6 | 0.1 0.1 0.6)
~ xmatrix=(0.3 | 0.2 0.3 | 0.2 0.2 0.3) cmatrix=(0 | 0 0 | 0 0 0)
New Line.drop bus1=x.1.2.3 bus2=h.1.2.3 phases=3 linecode=tpx length=0.1 units=kft
New Load.l120 bus1=h.1.3 phases=1 kV=0.12 kW=2 kvar=0.5 model=1
New Load.l240 bus1=h.1.2 phases=1 kV=0.24 kW=4 kvar=1 model=1
Set VoltageBases=[12.47 0.24]
CalcVoltageBases
""",
}


# Variant perturbations: four-wire lines on a km linecode with capacitance but lengths in m
# (a units trap of OpenDSS), a switch, a grounding reactor, a transformer neutral
# grounded through Rneut, movable single-phase loads (h1-h4), one on a single-phase lateral (h5)
# and one phase-to-phase (pp).
VARIANT_FEEDER = {
    "master.dss": """\
Clear
New Circuit.var basekv=11 pu=1.0 phases=3 bus1=mv
New Transformer.t windings=2 Buses=[mv.1.2.3 lv.1.2.3.4] Conns=[Delta Wye] kVs=[11 0.4]
~ kVAs=[400 400] XHL=4 %Rs=[0.5 0.5] Rneut=1 Xneut=0
New Linecode.lc4 nphases=4 units=km
~ rmatrix=(0.5 | 0.1 0.5 | 0.1 0.1 0.5 | 0.1 0.1 0.1 0.6)
~ xmatrix=(0.3 | 0.1 0.3 | 0.1 0.1 0.3 | 0.1 0.1 0.1 0.35)
~ cmatrix=(250 | -50 250 | -50 -50 250 | -40 -40 -40 220)
New Linecode.lc2 nphases=2 units=km
~ rmatrix=(0.8 | 0.1 0.8)
~ xmatrix=(0.35 | 0.1 0.35)
~ cmatrix=(200 | -40 200)
New Line.a bus1=lv.1.2.3.4 bus2=b1.1.2.3.4 phases=4 linecode=lc4 length=150 units=m
New Line.b bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 phases=4 linecode=lc4 length=80 units=m
New Line.sw bus1=b2.1.2.3.4 bus2=b3.1.2.3.4 phases=4 switch=yes
New Line.lat bus1=b2.2.4 bus2=b4.2.4 phases=2 linecode=lc2 length=40 units=m
New Reactor.grounding phases=1 bus1=b2.4 bus2=b2.0 R=10 X=0
New Load.h1 bus1=b1.1.4 phases=1 kV=0.23 kW=4 kvar=1 model=1
New Load.h2 bus1=b1.2.4 phases=1 kV=0.23 kW=3 kvar=1 model=1
New Load.h3 bus1=b2.3.4 phases=1 kV=0.23 kW=5 kvar=2 model=1
New Load.h4 bus1=b3.1.4 phases=1 kV=0.23 kW=2 kvar=0.5 model=1
New Load.h5 bus1=b4.2.4 phases=1 kV=0.23 kW=3 kvar=1 model=1
New Load.pp bus1=b3.1.2 phases=1 kV=0.4 kW=2 kvar=0.5 model=1
Set VoltageBases=[11 0.4]
CalcVoltageBases
""",
}


# 0.4 kV four-wire feeder in the scope of the OPF: lines, a neutral-grounding reactor at the
# source, single-phase phase-to-neutral loads and two PV units (``Generator.pv_<load>``) at the
# far end, 400 m from the source. At 30 kW each the PV raises the far end above 253 V.
OPF_FEEDER = {
    "master.dss": """\
Clear
New Circuit.opf basekv=0.4 pu=1.0 phases=3 bus1=src.1.2.3 angle=0 MVAsc3=100 MVAsc1=100
New Reactor.grounding phases=1 bus1=src.4 bus2=src.0 R=0 X=0.01
New Linecode.lc4 nphases=4 units=km
~ rmatrix=(0.5 | 0.1 0.5 | 0.1 0.1 0.5 | 0.1 0.1 0.1 0.6)
~ xmatrix=(0.3 | 0.1 0.3 | 0.1 0.1 0.3 | 0.1 0.1 0.1 0.35)
~ cmatrix=(0 | 0 0 | 0 0 0 | 0 0 0 0)
New Line.l1 bus1=src.1.2.3.4 bus2=b1.1.2.3.4 phases=4 linecode=lc4 length=0.2 units=km
New Line.l2 bus1=b1.1.2.3.4 bus2=b2.1.2.3.4 phases=4 linecode=lc4 length=0.2 units=km
New Load.h1 bus1=b1.1.4 phases=1 kV=0.23 kW=2 kvar=0.5 model=1
New Load.h2 bus1=b2.2.4 phases=1 kV=0.23 kW=2 kvar=0.5 model=1
New Load.h3 bus1=b2.3.4 phases=1 kV=0.23 kW=1 kvar=0.2 model=1
New Generator.pv_h2 bus1=b2.2.4 phases=1 kV=0.23 kW=2 kvar=0 model=1 vminpu=0.6 vmaxpu=1.5
New Generator.pv_h3 bus1=b2.3.4 phases=1 kV=0.23 kW=2 kvar=0 model=1 vminpu=0.6 vmaxpu=1.5
Set VoltageBases=[0.4]
CalcVoltageBases
""",
}


# Load profiles: an actual-value shape with a Q curve (a), a scaled shape without one (b,
# yearly), a one-point shape (c), a load without shape (d); multi-point shapes of 6 and 8 hours.
PROFILED = {
    "master.dss": """\
Clear
New Circuit.prof basekv=0.4 pu=1.0 phases=3 bus1=src
New Line.l1 bus1=src bus2=b1 phases=3 r1=0.05 x1=0.02 r0=0.1 x0=0.05 c1=0 c0=0 length=1 units=km
New Loadshape.actual npts=6 sinterval=3600 mult=(1 2 3 4 5 6) useactual=yes
~ qmult=(0.5 0.4 0.3 0.2 0.1 0)
New Loadshape.scaled npts=8 sinterval=3600 mult=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8) useactual=no
New Loadshape.flat npts=1 sinterval=3600 mult=(0.7) useactual=no
New Load.a bus1=b1.1 phases=1 kV=0.23 kW=1 kvar=1 model=1 daily=actual
New Load.b bus1=b1.2 phases=1 kV=0.23 kW=2 pf=0.9 model=1 yearly=scaled
New Load.c bus1=b1.3 phases=1 kV=0.23 kW=3 kvar=1 model=1 daily=flat
New Load.d bus1=b1.1 phases=1 kV=0.23 kW=1.5 kvar=0.5 model=1
Set VoltageBases=[0.4]
CalcVoltageBases
""",
}
