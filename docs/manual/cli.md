# Command-line reference

`gridfm_datakit_distr <command> --help` prints a command's options. In a uv checkout, prefix
`uv run` (`uv run --extra opf` for configs with OPF labels; always `--extra julia` once the `julia`
extra is installed, or uv removes it).

| Command | Does |
|---|---|
| [`version`](#version) | print the package version |
| [`fetch`](#fetch) | download catalog sources into the cache and verify them |
| [`inspect`](#inspect) | prepare one network and print its summary |
| [`validate`](#validate) | run the conformance suite on networks |
| [`generate`](#generate) | generate, resume or estimate a dataset |
| [`stats`](#stats) | statistics of a dataset |
| [`croissant`](#croissant) | rewrite and validate a dataset's Croissant metadata |
| [`export-posseq`](#export-posseq) | write the positive-sequence equivalent in gridfm-datakit's layout |
| [`setup-julia`](#setup-julia) | install and precompile the Julia environment of the cross-checks |

**Exit codes.** 0: done. 2: invalid input (config, unknown id, not a dataset, missing extra).
1: the work failed (a download, a check, a unit). An unexpected exception prints a traceback.

**Cache.** Network sources live in `$GRIDFM_DISTR_CACHE`, by default
`~/.cache/gridfm_datakit_distr`, one directory per source (PVGIS series in `pvgis/`); `fetch`,
`inspect`, `validate` and `generate` take `--cache DIR` instead. It is the only environment variable the package reads.

## `version`

Prints the version of the installed package.

## `fetch`

`fetch [ID...] [options]`: source ids (`epri_ieee`, `csiro_enwl`, `edp`) or network ids. A network
id fetches its whole source: `fetch ieee13` installs every IEEE feeder.

| Option | Effect |
|---|---|
| `--from-file PATH` | install from a local copy: a directory with the source's files, or the file itself for a single-file source; one source only; checksums are verified as for a download |
| `--allow-unverified` | accept files that have no checksum in the manifest; their SHA-256 are written to `<cache>/<source>.unverified.sha256` for review |
| `--config CONFIG` | also download the PVGIS series of a generation config with `scenarios.pv_profile.provider: pvgis`, one per location ([Scenarios](scenarios.md#pvgis-series)) |
| `--force` | download even if the source (or series) is cached |
| `--cache DIR` | cache directory |

Every file is checked against the source's SHA-256 manifest in the package
(`networks/manifests/`); archives are extracted in a staging directory that replaces the cached
copy only once everything is verified. The SHA-256 of every installed file, extracted ones
included, is kept in `<cache>/<source>/.fetch.sha256`. A source already fetched against the
current manifest is skipped once its files still match the manifest and that list; `fetch`,
`inspect`, `validate` and `generate` stop with an error naming the changed files otherwise, and
`fetch --force` restores them. EDP's RAR5 archives need `unrar`, `7z`, `7zz` or `bsdtar` on
`PATH`.

A PVGIS series cannot have a checksum in the package (the locations are the user's): `fetch
--config` records its SHA-256 at download next to it, in `<cache>/pvgis/`, and a cached series is
checked against it before each use; a changed series is an error, downloaded again only with
`--force`. The response must report the parameters requested (PVGIS ignores what it does not
parse). HTTP 429 and 529 (too many requests, overloaded) are retried three times, after 1, 2 and
4 s.

## `inspect`

`inspect NETWORK_ID [--passport-dir DIR] [--cache DIR]` compiles the network with its catalog
recipe, normalises it, solves its base case and prints its summary: licence, recipe, buses, nodes
and elements, neutral representation, voltage levels, power base, total load, normalisation,
base case and checks. `--passport-dir` also writes the passport to `<DIR>/<network_id>.json`.

## `validate`

`validate ID... [--out DIR] [--golden DIR] [--cache DIR]` runs the
[conformance suite](networks.md#conformance-suite). IDs are network ids, source ids (all their
networks) or `all` (every network whose source is in the cache). It prints one verdict per network
and writes `<out>/<network_id>.json` and `<out>/summary.md` (default `--out validation`).
Check C7 compares with golden files and is skipped without `--golden`; the repository holds them in
`integrationtests/golden` for the networks of `integrationtests/golden/networks.txt`, and a
network without a golden file fails C7. Exit 1 if a network fails.

## `generate`

`generate CONFIG [options]` writes the dataset of a YAML config to
`<settings.data_dir>/<dataset_name>` ([Configuration reference](config.md)).

| Option | Effect |
|---|---|
| `--dry-run` | prepare every network and print counts and estimates of time and storage; write nothing |
| `--data-dir DIR` | overrides `settings.data_dir` |
| `--processes N` | overrides `settings.num_processes` (default: one per CPU) |
| `--contexts N` | overrides `settings.contexts_per_process`: up to N units of one variant at once in each process, one thread and OpenDSS context each |
| `--overwrite` | sets `settings.overwrite` |
| `--cache DIR` | cache directory |

**What it checks first.** The config: every problem is listed as `<dotted.path>: <reason>`, exit
2. The sources of its networks must be in the cache (exit 1 otherwise). A loadshape provider
(`network_loadshapes`, `edp_smart_meter`) must find loadshapes in every network: the networks
without them are listed as `scenarios.profiles.provider: <network_id>: <reason>`, exit 2, also
with `--dry-run`. With `scenarios.pv_profile.provider: pvgis`, every network needs a location
(else an invalid config, exit 2) and its series in the cache, unchanged since download (else exit
1, naming the networks and `fetch --config`). Every network is prepared once before any work: one
that does not load stops the run. A resumed PVGIS dataset refuses a series whose SHA-256 differs
from the one in its manifest.

**New, resumed or refused.** The output directory is

- created when it does not exist;
- **resumed** when it holds a dataset with the same `config_hash`, the same versions (generator,
  OpenDSS, and CasADi with OPF labels) and the same schema version: units marked `done` in
  `manifest.json` are kept, all others run (failed ones again);
- **refused** when it holds a dataset of another config or version, unless `settings.overwrite`
  (`--overwrite`), which **deletes the directory** and starts again;
- refused, and left untouched, when it is not empty and has no `manifest.json`.

`config_hash` covers the whole resolved config except `data_dir`, `num_processes`,
`contexts_per_process` and `overwrite`. So the process and context counts can change between runs
of a dataset, and a dataset moved elsewhere resumes from its new `--data-dir`; nothing else can
change (not `chunk_size`, not the scenario count). Never run two `generate` commands on the same
directory at once.

**Output.** The units by status, the scenarios by power-flow status per network, the median and
95th percentile of the solve time, the size on disk, and the dataset's path. Exit 1, with the first
failure's message, if any unit failed; every unit's status and message is in `manifest.json`. With
`networks.neutral_representation: explicit`, a note lists the selected networks whose source
model has no neutral conductor.

## `stats`

`stats DATASET [--plots]` prints the summary of `generate` and, per network, the share of
scenarios with a voltage or thermal violation and percentiles of the phase voltages (pu of the
nominal line-to-neutral voltage), of the neutral voltage, of the voltage unbalance factor and of the
loadings. It writes them to `<dataset>/stats/stats.json`; `--plots` (needs the `plots` extra) also
writes one PNG per quantity to `<dataset>/stats/`. Voltage limits are the dataset's
`networks.voltage_limits`. Only converged scenarios count.

## `croissant`

`croissant DATASET [--validate/--no-validate]` rewrites `<dataset>/croissant.json` (MLCommons
Croissant 1.0; `generate` writes it too) and validates it with mlcroissant, which needs the
`croissant` extra ([Installation](../installation.md#optional-extras)). Exit 2 if the extra is
missing, 1 if the metadata are invalid.

## `export-posseq`

`export-posseq DATASET --out DIR [--network ID]... [--processes N]` builds the positive-sequence
equivalent of every converged scenario from the stored solutions and writes it in gridfm-datakit's
format to `<DIR>/<network_id>/raw` (gridfm-graphkit's `data_path`), with
`posseq_scenarios.parquet` (status of every scenario's view) and `posseq_mapping.parquet` next to
it ([Outputs §7.2](outputs.md#72-positive-sequence-export-export-posseq-not-stored-in-the-dataset)).
`--network` exports only that network (repeatable); `--processes` builds that many variants at a
time (default 1). A network with failed or unfinished units is refused (exit 2): resume `generate`
or leave it out with `--network`. Exit 1 if a view raised an error.

## `setup-julia`

Resolves, installs and precompiles the Julia environment (Julia and PowerModelsDistribution,
`gridfm_datakit_distr/juliapkg.json`) used only by the OPF cross-check tests; prints the versions
and timings. Needs the `julia` extra ([Installation](../installation.md#optional-extras)). OPF
labels do not use Julia.
