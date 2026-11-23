"""Tests for the executor: units, processes x contexts, manifest, resume."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet as pq
import pytest

from gridfm_datakit_distr.config.schema import GenerationConfig
from gridfm_datakit_distr.io import datakit_columns as datakit
from gridfm_datakit_distr.io import schema
from gridfm_datakit_distr.io.manifest import Manifest
from gridfm_datakit_distr.networks.catalog import ArchivesSource, Catalog, Network
from gridfm_datakit_distr.networks.fetch import ChecksumError
from gridfm_datakit_distr.solve import executor
from gridfm_datakit_distr.solve.executor import (
    GenerationError,
    generate,
    jobs_of,
    plan_tasks,
    plan_units,
)
from gridfm_datakit_distr.validate.balance import power_balance
from gridfm_datakit_distr.views.posseq_export import export_posseq
from tests.circuits import VARIANT_FEEDER, write_circuit

# A regulator bank with controllers (the states vary with tap jitter) and a lateral.
REGULATED = {
    "master.dss": """\
Clear
New Circuit.reg basekv=12.47 pu=1.0 phases=3 bus1=src angle=0 MVAsc3=200 MVAsc1=200
New Transformer.ra phases=1 windings=2 buses=[src.1 r.1] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[500 500] xhl=1 %rs=[0.1 0.1] bank=reg
New Transformer.rb phases=1 windings=2 buses=[src.2 r.2] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[500 500] xhl=1 %rs=[0.1 0.1] bank=reg
New Transformer.rc phases=1 windings=2 buses=[src.3 r.3] conns=[wye wye] kvs=[7.2 7.2]
~ kvas=[500 500] xhl=1 %rs=[0.1 0.1] bank=reg
New RegControl.ca transformer=ra winding=2 vreg=122 band=2 ptratio=60
New RegControl.cb transformer=rb winding=2 vreg=122 band=2 ptratio=60
New RegControl.cc transformer=rc winding=2 vreg=122 band=2 ptratio=60
New Line.l1 bus1=r.1.2.3 bus2=a.1.2.3 phases=3 r1=0.3 x1=0.6 r0=0.6 x0=1.2 c1=10 c0=5
~ length=2 units=km
New Line.lat bus1=a.1 bus2=x.1 phases=1 r1=0.4 x1=0.4 r0=0.4 x0=0.4 c1=0 c0=0
~ length=0.5 units=km
New Load.la bus1=a.1.2.3 phases=3 kV=12.47 kW=1500 kvar=500 model=1
New Load.lb bus1=a.2 phases=1 kV=7.2 kW=300 kvar=100 model=1
New Load.lx bus1=x.1 phases=1 kV=7.2 kW=200 kvar=50 model=1
Set VoltageBases=[12.47]
CalcVoltageBases
""",
}

CONFIG: dict[str, Any] = {
    "dataset_name": "d",
    "networks": {"include": ["t/*"]},
    "variants": {"n_variants": 2},
    "der": {"pv": {"penetration": 0.5}},
    "controls": {"tap_jitter_steps": 1},
    "scenarios": {"n_scenarios_per_variant": 5},
    "settings": {"seed": 11, "chunk_size": 2, "num_processes": 1},
}
TIMINGS = {"t_solve_s", "t_extract_s"}


def catalog(root: Path) -> Catalog:
    write_circuit(root / "src" / "feeder", VARIANT_FEEDER)
    write_circuit(root / "src" / "reg", REGULATED)
    source = ArchivesSource(
        kind="archives", id="src", title="test", homepage="-", license="CC0-1.0",
        citation="-", archives={},
    )  # fmt: skip
    networks = {
        "t/feeder": Network(id="t/feeder", source="src", master="feeder/master.dss"),
        "t/reg": Network(id="t/reg", source="src", master="reg/master.dss"),
    }
    return Catalog(sources={"src": source}, networks=networks)


def config(**settings: Any) -> GenerationConfig:
    data = copy.deepcopy(CONFIG)
    data["settings"].update(settings)
    return GenerationConfig.model_validate(data)


def files(root: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(root)): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }


def same_dataset(a: Path, b: Path) -> None:
    """Equal files but for the timing columns (and the manifest's execution settings)."""
    fa, fb = files(a), files(b)
    assert sorted(fa) == sorted(fb)
    for rel, pa_ in fa.items():
        if rel.endswith(".parquet"):
            ta, tb = pq.read_table(pa_), pq.read_table(fb[rel])
            drop = [c for c in ta.column_names if c in TIMINGS]
            assert ta.drop_columns(drop).equals(tb.drop_columns(drop)), rel
        else:
            assert pa_.read_bytes() == fb[rel].read_bytes(), rel
    ma, mb = Manifest.load(a), Manifest.load(b)
    assert ma.config_hash == mb.config_hash
    assert {k: u.status for k, u in ma.units.items()} == {k: u.status for k, u in mb.units.items()}


# ------------------------------------------------------------------------------------- planning


def test_units_and_tasks(tmp_path: Path) -> None:
    cfg = config()
    jobs = jobs_of(cfg, catalog(tmp_path))
    assert [(j.network_idx, j.network.id) for j in jobs] == [(0, "t/feeder"), (1, "t/reg")]
    units = plan_units(cfg, jobs)
    assert len(units) == 2 * 2 * 3  # networks x variants x chunks (2 + 2 + 1 scenarios)
    assert [(u.chunk, u.first, u.stop) for u in units[:3]] == [(0, 0, 2), (1, 2, 4), (2, 4, 5)]
    tasks = plan_tasks(units, jobs, 2)
    assert [len(t.units) for t in tasks] == [2, 1] * 4  # never across variants
    assert all(len({(u.network_idx, u.variant_idx) for u in t.units}) == 1 for t in tasks)


# ----------------------------------------------------------------------------------- generation


def test_a_dataset(tmp_path: Path) -> None:
    out = tmp_path / "out"
    manifest = generate(config(), out, catalog=catalog(tmp_path), root=tmp_path)
    assert {u.status for u in manifest.units.values()} == {"done"}
    assert all(u.peak_rss_mib is not None and u.peak_rss_mib > 50 for u in manifest.units.values())
    meta = pq.read_table(out / "scenarios" / "scenario_meta")
    ids = meta.column("scenario_id").to_pylist()
    assert len(ids) == len(set(ids)) == 2 * 2 * 5  # networks x variants x scenarios
    assert not (out / "states" / "states").exists()  # parts merged at finalisation
    states = pq.read_table(out / "states" / "states.parquet")
    with_state = [s for s in meta.column("state_id").to_pylist() if s is not None]
    assert sum(states.column("n_scenarios").to_pylist()) == len(with_state)
    assert len(set(states.column("state_id").to_pylist())) == states.num_rows
    # the origin is per scenario: scenarios that share a state can reach it by different routes
    assert "origin" not in states.column_names
    rows = meta.select(["network_id", "state_id", "origin"]).to_pylist()
    assert all((r["state_id"] is None) == (r["origin"] is None) for r in rows)
    reg = {r["origin"] for r in rows if r["network_id"] == "t/reg" and r["origin"]}
    feeder = {r["origin"] for r in rows if r["network_id"] == "t/feeder" and r["origin"]}
    assert "static_settle+jitter" in reg
    assert feeder == {"static_settle"}  # no regulator to jitter
    assert (out / "variants" / "semantic" / "t" / "reg" / "v0001.json").exists()
    nodes = pq.read_table(out / "variants" / "nodes")
    buses, backbone = nodes.column("bus").to_pylist(), nodes.column("backbone_bus").to_pylist()
    owner = dict(zip(buses, backbone, strict=True))
    assert owner["b4"] == "b2"  # VARIANT_FEEDER's lateral
    assert owner["x"] == "a"  # REGULATED's lateral
    assert not (out / "views").exists()  # the truth only
    assert not (out / "scenarios" / "element_results").exists()  # Yprim · V (labels default)
    exports = export_posseq(out, tmp_path / "export", processes=2)
    assert [r.network_id for r in exports] == ["t/feeder", "t/reg"]
    for r in exports:
        raw = r.raw_dir
        assert r.scenarios == 2 * 5
        assert r.statuses == {"converged": 10}
        assert (raw / "n_scenarios.txt").read_text(encoding="utf-8") == "10"
        assert pq.read_table(raw.parent / "posseq_scenarios.parquet").num_rows == 10
        assert pq.read_table(raw.parent / "posseq_mapping.parquet").num_rows > 0
        assert not (raw.parent / ".posseq-work").exists()
        tables = {}
        for kind, name in datakit.FILES.items():
            written = pq.read_table(raw / name / "scenario_partition=0")
            assert written.column_names == ["scenario", *schema.DATAKIT_TABLES[kind].names]
            scenario = written.column("scenario").to_numpy()
            assert scenario.min() == 0  # graphkit's assertion: 0 .. n - 1
            assert scenario.max() == len(np.unique(scenario)) - 1
            tables[kind] = written
        balance = power_balance(tables["bus"], tables["branch"])
        assert balance.max() < 1e-6  # MW: datakit's power balance on the exported tables
    assert json.loads((out / "catalog" / "passports" / "t" / "reg.json").read_text())
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("# d\n")


def test_the_same_dataset_with_processes_and_contexts(tmp_path: Path) -> None:
    """The same dataset with (1 process, 1 context) and (4 processes, 2 contexts)."""
    cat = catalog(tmp_path)
    generate(config(), tmp_path / "a", catalog=cat, root=tmp_path)
    parallel = config(num_processes=4, contexts_per_process=2)
    generate(parallel, tmp_path / "b", catalog=cat, root=tmp_path)
    same_dataset(tmp_path / "a", tmp_path / "b")


def test_an_interrupted_run_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No duplicates, and the dataset of an uninterrupted run."""
    cat = catalog(tmp_path)
    generate(config(), tmp_path / "full", catalog=cat, root=tmp_path)
    real = executor._run_unit
    calls: list[str] = []

    def crashing(*args: Any) -> Any:
        if len(calls) == 5:
            raise KeyboardInterrupt
        calls.append(args[2].key)
        return real(*args)

    monkeypatch.setattr(executor, "_run_unit", crashing)
    out = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        generate(config(), out, catalog=cat, root=tmp_path)
    done = {k for k, u in Manifest.load(out).units.items() if u.status == "done"}
    assert done == set(calls)
    monkeypatch.undo()
    rerun: list[str] = []

    def counting(*args: Any) -> Any:
        rerun.append(args[2].key)
        return real(*args)

    monkeypatch.setattr(executor, "_run_unit", counting)
    generate(config(), out, catalog=cat, root=tmp_path)
    assert set(rerun).isdisjoint(done)  # done units are not run again
    same_dataset(tmp_path / "full", out)


def test_an_interrupted_pool_stops_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C with several processes: the queued tasks are cancelled, not run unrecorded."""
    cat = catalog(tmp_path)
    cfg = config(num_processes=2, chunk_size=1)  # 20 units, one task each
    real = executor._record

    def interrupted(*args: Any) -> None:
        real(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(executor, "_record", interrupted)
    out = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        generate(cfg, out, catalog=cat, root=tmp_path)
    ran = list((out / "scenarios" / "scenario_meta").glob("part-*.parquet"))
    assert 1 <= len(ran) < 20
    monkeypatch.undo()
    generate(cfg, out, catalog=cat, root=tmp_path)
    generate(cfg, tmp_path / "full", catalog=cat, root=tmp_path)
    same_dataset(tmp_path / "full", out)


def test_another_config_needs_overwrite(tmp_path: Path) -> None:
    cat = catalog(tmp_path)
    out = tmp_path / "out"
    generate(config(), out, catalog=cat, root=tmp_path)
    with pytest.raises(GenerationError, match="another config"):
        generate(config(seed=12), out, catalog=cat, root=tmp_path)
    manifest = generate(config(seed=12, overwrite=True), out, catalog=cat, root=tmp_path)
    assert manifest.config["settings"]["seed"] == 12
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.txt").write_text("mine")
    with pytest.raises(GenerationError, match="not a dataset"):
        generate(config(), stray, catalog=cat, root=tmp_path)
    assert (stray / "notes.txt").read_text() == "mine"


def test_the_identity_holds_casadi_only_with_opf_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another CasADi refuses a resume, like another OpenDSS; without OPF labels it is not read."""
    cat = catalog(tmp_path)
    on = GenerationConfig.model_validate({**CONFIG, "labels": {"opf": {"enabled": True}}})
    monkeypatch.setattr(executor, "version", lambda name: "1.0")
    assert executor.versions(on)["casadi"] == "1.0"
    monkeypatch.setattr(executor, "version", lambda name: 1 / 0)  # any read would fail
    assert "casadi" not in executor.versions(config())
    out, jobs = tmp_path / "out", jobs_of(on, cat)
    monkeypatch.setattr(executor, "version", lambda name: "1.0")
    executor._open(on, out, jobs)
    executor._open(on, out, jobs)  # the same versions resume
    monkeypatch.setattr(executor, "version", lambda name: "2.0")
    with pytest.raises(GenerationError, match="another config or version"):
        executor._open(on, out, jobs)


def test_failed_units_are_recorded_without_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = executor._unit_tables

    def failing(slot: Any, task: Any, unit: Any, cfg: Any, digest: str) -> Any:
        if task.job.network.id == "t/reg":
            raise RuntimeError("boom")
        return real(slot, task, unit, cfg, digest)

    cat, out = catalog(tmp_path), tmp_path / "out"
    generate(config(), out, catalog=cat, root=tmp_path)
    # t/reg's files are on disk but its units unrecorded: an attempt interrupted before the manifest
    crashed = Manifest.load(out)
    for u in crashed.units.values():
        if u.network_id == "t/reg":
            crashed.finish(u, "pending")
    monkeypatch.setattr(executor, "_unit_tables", failing)
    manifest = generate(config(), out, catalog=cat, root=tmp_path)
    failed = [u for u in manifest.units.values() if u.status == "failed"]
    assert {u.network_id for u in failed} == {"t/reg"}
    assert all("RuntimeError: boom" in (u.message or "") for u in failed)
    assert not list(out.rglob("part-00001-*"))  # t/reg is network 1: no unit file of any attempt
    assert not list((out / "variants" / "semantic" / "t").glob("reg/*"))
    assert (out / "README.md").exists()
    with pytest.raises(ValueError, match=r"incomplete.*t/reg"):
        export_posseq(out, tmp_path / "export")
    [result] = export_posseq(out, tmp_path / "export", ["t/feeder"])  # complete: exported
    assert result.statuses == {"converged": 10}


def test_a_dataset_of_an_earlier_config_is_exported(tmp_path: Path) -> None:
    """``export-posseq`` reads a manifest that holds keys removed since it was written."""
    data = copy.deepcopy(CONFIG)
    data["networks"]["include"] = ["t/feeder"]
    data["variants"]["n_variants"] = 1
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(data), out, catalog=catalog(tmp_path), root=tmp_path)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["config"]["labels"]["pf"] = True
    manifest["config"]["normalisation"]["loads"]["model"] = "as_is"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    [result] = export_posseq(out, tmp_path / "export")
    assert result.statuses == {"converged": 5}


def test_a_changed_cached_network_stops_the_generation(tmp_path: Path) -> None:
    """``generate`` re-checks the files that ``fetch`` installed before it writes anything."""
    cat = catalog(tmp_path)
    (tmp_path / "src" / ".fetch.json").write_text("{}")
    # what fetch recorded differs from the file on disk
    (tmp_path / "src" / ".fetch.sha256").write_text(f"{'0' * 64}  feeder/master.dss\n")
    with pytest.raises(ChecksumError, match=r"feeder/master\.dss"):
        generate(config(), tmp_path / "out", catalog=cat, root=tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("network_id", ["../decoy", "/abs/decoy", ""])
def test_export_refuses_a_network_id_that_leaves_the_output(
    tmp_path: Path, network_id: str
) -> None:
    """A dataset is outside input: its network ids become folders under ``out``, never outside."""
    manifest = Manifest.create(tmp_path / "dataset", {}, {})
    manifest.networks = [{"network_id": network_id, "network_idx": 0, "representation": None}]
    manifest.save()
    decoy = tmp_path / "decoy" / "raw"
    decoy.mkdir(parents=True)
    with pytest.raises(ValueError, match="unsafe"):
        export_posseq(tmp_path / "dataset", tmp_path / "export")
    assert decoy.exists()


def test_a_configured_power_base_is_the_datasets(tmp_path: Path) -> None:
    """A number for ``normalisation.s_base_va`` is every network's power base: in
    ``catalog/networks``, the passports and the semantic layer."""
    data = copy.deepcopy(CONFIG)
    data["networks"]["include"] = ["t/feeder"]
    data["variants"]["n_variants"] = 1
    data["normalisation"] = {"s_base_va": 2.5e5}
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(data), out, catalog=catalog(tmp_path), root=tmp_path)
    networks = pq.read_table(out / "catalog" / "networks.parquet")
    assert networks.column("s_base_va").to_pylist() == [2.5e5]
    passport = json.loads((out / "catalog" / "passports" / "t" / "feeder.json").read_text())
    assert (passport["s_base_va"], passport["s_base_origin"]) == (2.5e5, "configured")
    semantic = out / "variants" / "semantic" / "t" / "feeder" / "v0000.json"
    assert json.loads(semantic.read_text())["settings"]["sbase_default"] == 250.0


def test_a_network_without_a_view_exports_its_errors(tmp_path: Path) -> None:
    """A single-phase network has no three-phase backbone: it generates, its nodes have no
    ``backbone_bus``, and the export records the error of every scenario's view."""
    cat = catalog(tmp_path)
    write_circuit(
        tmp_path / "src" / "one",
        {
            "master.dss": "Clear\nNew Circuit.one basekv=12.47 pu=1.0 phases=3 bus1=src\n"
            "New Line.l1 bus1=src.1 bus2=a.1 phases=1 r1=0.3 x1=0.6 r0=0.3 x0=0.6 c1=0 c0=0\n"
            "New Load.ld bus1=a.1 phases=1 kV=7.2 kW=200 kvar=50 model=1\n"
            "Set VoltageBases=[12.47]\nCalcVoltageBases\n"
        },
    )
    cat = Catalog(
        sources=cat.sources,
        networks={"t/one": Network(id="t/one", source="src", master="one/master.dss")},
    )
    out = tmp_path / "out"
    manifest = generate(config(), out, catalog=cat, root=tmp_path)
    assert {u.status for u in manifest.units.values()} == {"done"}
    assert set(pq.read_table(out / "variants" / "nodes").column("backbone_bus").to_pylist()) == {
        None
    }
    stale = tmp_path / "export" / "t" / "one" / "raw"
    stale.mkdir(parents=True)  # an earlier export's tables
    [result] = export_posseq(out, tmp_path / "export")
    assert (result.scenarios, result.statuses) == (0, {"error": 10})
    assert not stale.exists()
    status = pq.read_table(tmp_path / "export" / "t" / "one" / "posseq_scenarios.parquet")
    assert all("PosSeqError" in m for m in status.column("view_message").to_pylist())


def test_a_network_draws_the_same_whatever_else_the_dataset_holds(tmp_path: Path) -> None:
    """t/reg alone and after t/feeder: equal tables but for the network bits of ``scenario_id``
    and the part names."""
    both, alone = tmp_path / "both", tmp_path / "alone"
    networks = catalog(tmp_path)
    generate(config(), both, catalog=networks, root=tmp_path)
    data = copy.deepcopy(CONFIG)
    data["networks"]["include"] = ["t/reg"]
    generate(GenerationConfig.model_validate(data), alone, catalog=networks, root=tmp_path)
    mask = (1 << 47) - 1

    def local(table: pa.Table) -> pa.Table:
        # config_hash identifies the dataset's config, which differs by its network list
        drop = [c for c in table.column_names if c in TIMINGS | {"config_hash"}]
        table = table.drop_columns(drop)
        if "scenario_id" in table.column_names:
            sids = table.column("scenario_id").to_pylist()
            ids = pa.array([s & mask for s in sids], pa.int64())
            table = table.set_column(
                table.schema.get_field_index("scenario_id"), "scenario_id", ids
            )
        return table

    parts = sorted(both.rglob("part-00001-*.parquet"))
    assert parts
    for part in parts:
        rel = part.relative_to(both)
        twin = alone / rel.parent / part.name.replace("part-00001-", "part-00000-")
        assert local(pq.read_table(part)).equals(local(pq.read_table(twin))), rel
    states = pq.read_table(alone / "states" / "states.parquet")
    mine = set(states.column("state_id").to_pylist())
    for name in ("states/states", "states/ybus"):
        table = pq.read_table(both / f"{name}.parquet")
        table = table.filter(pa.compute.is_in(table["state_id"], pa.array(sorted(mine))))
        assert table.equals(pq.read_table(alone / f"{name}.parquet")), name


def test_storage_enters_the_exported_der_columns(tmp_path: Path) -> None:
    """The positive-sequence export reads every ``scenarios/der`` row: storage at its set-point
    in ``x_der_p_mw`` and ``x_der_p_avail_mw``."""
    data = copy.deepcopy(CONFIG)
    data["networks"]["include"] = ["t/feeder"]
    data["variants"]["n_variants"] = 1
    data["der"] = {"pv": {"penetration": 1.0}, "storage": {"share": 1.0}}
    out = tmp_path / "out"
    generate(GenerationConfig.model_validate(data), out, catalog=catalog(tmp_path), root=tmp_path)
    export_posseq(out, tmp_path / "export")
    der = pq.read_table(out / "scenarios" / "der").to_pylist()
    assert {r["der_type"] for r in der} == {"pv", "bess"}
    expected: dict[int, list[float]] = {}
    for r in der:
        e = expected.setdefault(r["scenario_id"], [0.0, 0.0])
        e[0] += r["p_w"] / 1e6
        e[1] += r["p_avail_w"] / 1e6
    parts = [
        pq.read_table(p)
        for p in (tmp_path / "export").rglob("*.parquet")
        if p.is_file() and "x_der_p_mw" in pq.read_schema(p).names
    ]
    assert parts
    got: dict[int, list[float]] = {}
    for t in parts:
        for r in t.select(["x_scenario_id", "x_der_p_mw", "x_der_p_avail_mw"]).to_pylist():
            g = got.setdefault(r["x_scenario_id"], [0.0, 0.0])
            g[0] += r["x_der_p_mw"]
            g[1] += r["x_der_p_avail_mw"]
    assert got
    for sid, (p, avail) in got.items():
        assert p == pytest.approx(expected[sid][0], abs=1e-9)
        assert avail == pytest.approx(expected[sid][1], abs=1e-9)
