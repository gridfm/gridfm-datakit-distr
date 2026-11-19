"""Command-line interface: ``gridfm_datakit_distr <command>``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from gridfm_datakit_distr import __version__

if TYPE_CHECKING:
    from gridfm_datakit_distr.config.schema import GenerationConfig
    from gridfm_datakit_distr.solve.executor import Job

app = typer.Typer(
    name="gridfm_datakit_distr",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Generate PF/OPF datasets for three-phase unbalanced distribution networks."""
    # An explicit callback keeps Typer in sub-command mode even with a single command.


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fetch(
    idents: Annotated[
        list[str], typer.Argument(help="Source or network ids, e.g. ieee13, csiro_enwl, edp.")
    ],
    from_file: Annotated[
        Path | None,
        typer.Option(
            "--from-file",
            help="Install from a local copy: a directory with the source's files "
            "(or the file itself for single-file sources). Checksums are verified.",
        ),
    ] = None,
    allow_unverified: Annotated[
        bool,
        typer.Option(
            "--allow-unverified",
            help="Accept files without a manifest checksum and write their SHA-256 to "
            "<cache>/<source>.unverified.sha256 for review.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Download even if cached.")] = False,
    cache: Annotated[
        Path | None,
        typer.Option(
            "--cache", help="Cache directory (default: $GRIDFM_DISTR_CACHE or ~/.cache/...)."
        ),
    ] = None,
) -> None:
    """Download catalog sources into the cache and verify their checksums."""
    from gridfm_datakit_distr.networks.catalog import load_catalog
    from gridfm_datakit_distr.networks.fetch import FetchError, fetch_source

    catalog = load_catalog()
    try:
        sources = {src.id: src for src in (catalog.source_for(i) for i in idents)}
    except KeyError as exc:
        typer.echo(f"error: {exc.args[0]}", err=True)
        raise typer.Exit(2) from exc
    if from_file is not None and len(sources) != 1:
        typer.echo("error: --from-file needs exactly one source", err=True)
        raise typer.Exit(2)

    for source in sources.values():
        try:
            result = fetch_source(
                source,
                root=cache,
                from_path=from_file,
                allow_unverified=allow_unverified,
                force=force,
            )
        except FetchError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if result.skipped:
            typer.echo(f"{source.id}: already fetched and verified ({result.directory})")
            continue
        typer.echo(f"{source.id}: {result.n_files} files installed in {result.directory}")
        if result.lock_file is not None:
            typer.echo(
                f"{source.id}: {len(result.unverified)} files have no manifest checksum; "
                f"review {result.lock_file} and add it to networks/manifests/{source.id}.sha256"
            )


@app.command()
def inspect(
    network_id: Annotated[str, typer.Argument(help="Network id, e.g. ieee13, csiro_enwl/n02_f5.")],
    passport_dir: Annotated[
        Path | None,
        typer.Option("--passport-dir", help="Also write the passport to <dir>/<network_id>.json."),
    ] = None,
    cache: Annotated[
        Path | None,
        typer.Option(
            "--cache", help="Cache directory (default: $GRIDFM_DISTR_CACHE or ~/.cache/...)."
        ),
    ] = None,
) -> None:
    """Prepare a network (recipe, normalisation, base case) and print its summary."""
    from gridfm_datakit_distr.networks.cache import is_fetched
    from gridfm_datakit_distr.networks.catalog import load_catalog
    from gridfm_datakit_distr.networks.passport import build_passport, inspect_text, write_passport
    from gridfm_datakit_distr.networks.prepare import prepare_network

    catalog = load_catalog()
    try:
        network = catalog.network(network_id)
    except KeyError as exc:
        typer.echo(f"error: {exc.args[0]}", err=True)
        raise typer.Exit(2) from exc
    if not is_fetched(catalog.source(network.source), cache):
        typer.echo(f"error: run `gridfm_datakit_distr fetch {network.source}` first", err=True)
        raise typer.Exit(1)
    passport = build_passport(prepare_network(network, root=cache), catalog)
    typer.echo(inspect_text(passport), nl=False)
    if passport_dir is not None:
        typer.echo(f"passport: {write_passport(passport, passport_dir)}")


@app.command()
def validate(
    idents: Annotated[
        list[str],
        typer.Argument(
            help="Network ids, source ids (all their networks) or 'all' (every "
            "network whose source is in the cache)."
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Directory for <network_id>.json and summary.md.")
    ] = Path("validation"),
    golden: Annotated[
        Path | None,
        typer.Option("--golden", help="Golden-file directory for check C7 (a missing file fails)."),
    ] = None,
    cache: Annotated[
        Path | None,
        typer.Option(
            "--cache", help="Cache directory (default: $GRIDFM_DISTR_CACHE or ~/.cache/...)."
        ),
    ] = None,
) -> None:
    """Run the conformance suite (C1-C7, bases, setpoints) and write the reports."""
    from gridfm_datakit_distr.networks.cache import is_fetched
    from gridfm_datakit_distr.networks.catalog import load_catalog
    from gridfm_datakit_distr.validate.report import write_reports
    from gridfm_datakit_distr.validate.suite import validate_network

    catalog = load_catalog()
    fetched = {s for s in catalog.sources if is_fetched(catalog.source(s), cache)}
    ids: list[str] = []
    for ident in idents:
        if ident == "all":
            ids += [n for n, net in catalog.networks.items() if net.source in fetched]
        elif ident in catalog.sources:
            ids += [n for n, net in catalog.networks.items() if net.source == ident]
        elif ident in catalog.networks:
            ids.append(ident)
        else:
            typer.echo(f"error: unknown network or source {ident!r}", err=True)
            raise typer.Exit(2)
    ids = list(dict.fromkeys(ids))
    missing = sorted({catalog.network(i).source for i in ids} - fetched)
    if missing:
        typer.echo(f"error: sources not fetched: {', '.join(missing)}", err=True)
        raise typer.Exit(1)
    if not ids:
        typer.echo("error: no network to validate (nothing fetched?)", err=True)
        raise typer.Exit(1)
    results = []
    for network_id in ids:
        result = validate_network(catalog.network(network_id), root=cache, golden_dir=golden)
        results.append(result)
        failures = [f"{c.id}: {c.detail}" for c in result.checks if c.status == "fail"]
        verdict = "pass" if result.passed else f"FAIL {result.error or '; '.join(failures)}"
        typer.echo(f"{network_id}: {verdict} ({result.seconds:.1f} s)")
    summary = write_reports(results, out)
    n_failed = sum(not r.passed for r in results)
    typer.echo(f"{len(results) - n_failed}/{len(results)} pass; summary: {summary}")
    if n_failed:
        raise typer.Exit(1)


@app.command()
def generate(
    config_path: Annotated[
        Path, typer.Argument(help="Generation config (YAML); keys in docs/manual/config.md.")
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print counts and time/storage estimates; write nothing."),
    ] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Overrides settings.data_dir.")
    ] = None,
    processes: Annotated[
        int | None,
        typer.Option("--processes", min=1, help="Overrides settings.num_processes."),
    ] = None,
    contexts: Annotated[
        int | None,
        typer.Option("--contexts", min=1, help="Overrides settings.contexts_per_process."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite", help="Replace a dataset of another config (settings.overwrite)."
        ),
    ] = False,
    cache: Annotated[
        Path | None,
        typer.Option(
            "--cache", help="Cache directory (default: $GRIDFM_DISTR_CACHE or ~/.cache/...)."
        ),
    ] = None,
) -> None:
    """Generate a dataset from a config, or resume it (same config); --dry-run estimates it."""
    from gridfm_datakit_distr.config.load import ConfigError, load_config
    from gridfm_datakit_distr.networks.cache import is_fetched
    from gridfm_datakit_distr.networks.catalog import load_catalog
    from gridfm_datakit_distr.solve.executor import GenerationError, jobs_of
    from gridfm_datakit_distr.solve.executor import generate as run_generation

    catalog = load_catalog()
    try:
        config = load_config(config_path, catalog)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    updates: dict[str, object] = {}
    if data_dir is not None:
        updates["data_dir"] = str(data_dir)
    if processes is not None:
        updates["num_processes"] = processes
    if contexts is not None:
        updates["contexts_per_process"] = contexts
    if overwrite:
        updates["overwrite"] = True
    if updates:  # execution settings only: not part of the config hash
        config = config.model_copy(update={"settings": config.settings.model_copy(update=updates)})
    jobs = jobs_of(config, catalog)
    if config.networks.neutral_representation == "explicit":
        kron = [
            j.network.id for j in jobs if j.representation is None and j.network.neutral == "kron"
        ]
        if kron:  # `explicit` loads each network's own model: say when that has no neutral
            typer.echo(
                f"note: no explicit neutral in {', '.join(kron)} (Kron-reduced source model)"
            )
    missing = sorted(
        {j.network.source for j in jobs if not is_fetched(catalog.source(j.network.source), cache)}
    )
    if missing:
        typer.echo(f"error: run `gridfm_datakit_distr fetch {' '.join(missing)}` first", err=True)
        raise typer.Exit(1)
    out = Path(config.settings.data_dir) / config.dataset_name
    if dry_run:
        typer.echo(_dry_run(config, jobs, cache, out), nl=False)
        return
    try:
        manifest = run_generation(config, out, catalog=catalog, root=cache)
    except GenerationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    failed = [u for u in manifest.units.values() if u.status != "done"]
    from gridfm_datakit_distr.stats.compute import summary, summary_text

    typer.echo(summary_text(summary(out)), nl=False)
    typer.echo(f"dataset: {out}")
    if failed:
        typer.echo(f"error: {len(failed)} unit(s) failed, e.g. {failed[0].message}", err=True)
        raise typer.Exit(1)


def _dry_run(config: GenerationConfig, jobs: list[Job], cache: Path | None, out: Path) -> str:
    """Counts and estimates of a generation (:mod:`solve.estimate`)."""
    from gridfm_datakit_distr.networks.prepare import prepare_network
    from gridfm_datakit_distr.solve.estimate import CONTROLS_FACTOR, estimate_network
    from gridfm_datakit_distr.solve.executor import plan_units
    from gridfm_datakit_distr.solve.pf_batch import prepare_options

    options = prepare_options(config)
    n_variants = config.variants.n_variants
    n_scenarios = config.scenarios.n_scenarios_per_variant
    units = plan_units(config, jobs)
    lines = [
        f"dataset {config.dataset_name} -> {out}",
        f"{len(jobs)} network(s) x {n_variants} variant(s) x {n_scenarios} scenarios = "
        f"{len(jobs) * n_variants * n_scenarios} scenarios in {len(units)} units of at most "
        f"{config.settings.chunk_size}",
        "network: nodes, customers; estimated per scenario (one context)",
    ]
    seconds = size = 0.0
    controls = []
    for job in jobs:
        e = estimate_network(prepare_network(job.network, options, cache), config)
        n = n_variants * n_scenarios
        seconds += e.seconds_per_scenario * n
        size += e.bytes_per_scenario * n + e.bytes_per_variant * n_variants
        flag = " (controls: may take several times longer)" if e.controls else ""
        if e.controls:
            controls.append(job.network.id)
        lines.append(
            f"  {job.network.id}: {e.nodes} nodes, {e.customers} customers; "
            f"{e.seconds_per_scenario * 1e3:.1f} ms, {e.bytes_per_scenario / 1e3:.1f} kB{flag}"
        )
    processes = config.settings.num_processes or os.cpu_count() or 1
    lines += [
        f"estimated CPU time {_duration(seconds)}, about {_duration(seconds / processes)} of wall "
        f"time on {processes} process(es); storage {_size(size)}",
        "estimates from measurements: order of magnitude only"
        + (
            f"; networks with controls measured up to {CONTROLS_FACTOR:g}x the time estimate"
            if controls
            else ""
        ),
    ]
    return "\n".join(lines) + "\n"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _size(n_bytes: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {unit}"
    return f"{n_bytes:.0f} B"


@app.command()
def stats(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory (with manifest.json).")],
    plots: Annotated[
        bool, typer.Option("--plots", help="Also write PNG figures (needs the 'plots' extra).")
    ] = False,
) -> None:
    """Statistics of a generated dataset: summary, voltages, neutral voltage, unbalance (VUF),
    loadings and violation rates; writes <dataset>/stats/stats.json."""
    from gridfm_datakit_distr.io.manifest import MANIFEST, Manifest
    from gridfm_datakit_distr.stats import plots as figures
    from gridfm_datakit_distr.stats.compute import summary, summary_text
    from gridfm_datakit_distr.stats.physics import physics, report_text, write_report

    if not (dataset / MANIFEST).exists():
        typer.echo(f"error: {dataset} is not a dataset (no {MANIFEST})", err=True)
        raise typer.Exit(2)
    if plots and not figures.available():
        typer.echo("error: --plots needs matplotlib: install the 'plots' extra", err=True)
        raise typer.Exit(2)
    typer.echo(summary_text(summary(dataset)), nl=False)
    limits = Manifest.load(dataset).config["networks"]["voltage_limits"]
    vmin, vmax = float(limits["vmin_pu"]), float(limits["vmax_pu"])
    found = physics(dataset, vmin, vmax)
    typer.echo(report_text(found), nl=False)
    typer.echo(f"statistics: {write_report(dataset, found, vmin, vmax)}")
    if plots:
        for path in figures.write_plots(dataset, found):
            typer.echo(f"figure: {path}")


@app.command()
def croissant(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory (with manifest.json).")],
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate with mlcroissant (needs the 'croissant' extra).",
        ),
    ] = True,
) -> None:
    """Write <dataset>/croissant.json (Croissant 1.0 metadata) and validate it."""
    from gridfm_datakit_distr.io import croissant as metadata
    from gridfm_datakit_distr.io.manifest import MANIFEST

    if not (dataset / MANIFEST).exists():
        typer.echo(f"error: {dataset} is not a dataset (no {MANIFEST})", err=True)
        raise typer.Exit(2)
    path = metadata.write_croissant(dataset)
    typer.echo(f"croissant: {path}")
    if not validate:
        return
    try:
        warnings = metadata.validate(path)
    except ImportError as exc:
        typer.echo("error: validation needs mlcroissant: install the 'croissant' extra", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:  # mlcroissant.ValidationError: the report lists the errors
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for w in warnings:
        typer.echo(f"warning: {w}")
    typer.echo("valid (mlcroissant)")


@app.command("export-posseq")
def export_posseq(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory (with manifest.json).")],
    out: Annotated[
        Path, typer.Option("--out", help="Directory for <network_id>/raw (graphkit's data_path).")
    ],
    networks: Annotated[
        list[str] | None, typer.Option("--network", help="Only this network (repeatable).")
    ] = None,
    processes: Annotated[
        int, typer.Option("--processes", min=1, help="Processes (one variant each at a time).")
    ] = 1,
) -> None:
    """Export the positive-sequence equivalent of a dataset in gridfm-datakit's layout (the input
    of the balanced GENCO), computed from the stored three-phase solutions; one directory per
    network. Exit 1 if a view raised an error."""
    from gridfm_datakit_distr.io.manifest import MANIFEST
    from gridfm_datakit_distr.views.posseq_export import export_posseq as run_export

    if not (dataset / MANIFEST).exists():
        typer.echo(f"error: {dataset} is not a dataset (no {MANIFEST})", err=True)
        raise typer.Exit(2)
    try:
        results = run_export(dataset, out, networks, processes)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    errors = 0
    for r in results:
        views = ", ".join(f"{status} {n}" for status, n in r.statuses.items()) or "none"
        where = f" in {r.raw_dir}" if r.scenarios else ""
        typer.echo(f"{r.network_id}: {r.scenarios} scenarios{where} (views: {views})")
        errors += r.statuses.get("error", 0)
    if errors:
        typer.echo(f"error: {errors} views raised; see posseq_scenarios.parquet", err=True)
        raise typer.Exit(1)


@app.command("setup-julia")
def setup_julia() -> None:
    """Install and precompile the Julia environment for OPF (needs the 'julia' extra)."""
    from gridfm_datakit_distr.opf.julia_env import setup

    for key, value in setup().items():
        shown = f"{value:.1f} s" if isinstance(value, float) else value
        typer.echo(f"{key}: {shown}")
