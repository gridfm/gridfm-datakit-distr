# Installation

You need Python 3.10, 3.11 or 3.12. The power flow runs in OpenDSS through
[DSS-Extensions](https://dss-extensions.org/) (OpenDSSDirect.py), installed from PyPI with the
package.

## From the repository

```bash
pip install "gridfm-datakit-distr @ git+ssh://git@github.com/cardef/gridfm-datakit-distr.git@v0.1.0"
gridfm_datakit_distr version
```

The repository is private for now: the command needs access to it.

## For developers

With [uv](https://docs.astral.sh/uv/), which creates `.venv` with the versions pinned in `uv.lock`:

```bash
git clone git@github.com:cardef/gridfm-datakit-distr.git
cd gridfm-datakit-distr
uv sync
uv run pytest -m "not slow and not julia"
```

Tests are in `tests/` (unit) and `integrationtests/` (catalog networks, golden files,
benchmarks). Markers: `slow` (> 10 s), `network_data` (needs fetched networks), `perf`
(benchmarks, run with `-m perf`), `julia` (needs Julia, run with `DISTR_RUN_JULIA=1`),
`conformance`. Lint and types: `uv run ruff check && uv run ruff format --check && uv run mypy`.

## Network data

Networks are never stored in the repository. `fetch` downloads a catalog source into the cache
(`$GRIDFM_DISTR_CACHE`, default `~/.cache/gridfm_datakit_distr`) and verifies every file against
its SHA-256 manifest:

```bash
gridfm_datakit_distr fetch ieee13 csiro_enwl     # network or source ids
gridfm_datakit_distr fetch edp                   # RAR archives: needs unrar, 7z, 7zz or bsdtar
```

## Optional extras

| Extra | Install | For |
|---|---|---|
| `plots` | `pip install "gridfm-datakit-distr[plots] @ …"` | `stats --plots` |
| `croissant` | in its own environment | `croissant <dataset>`: validation with mlcroissant, whose type stubs break the project's type checking |
| `julia` | `uv sync --extra julia`, then `uv run --extra julia gridfm_datakit_distr setup-julia` | PowerModelsDistribution for OPF and cross-checks (Julia 1.10, about 1 GB) |
| `docs` | `uv sync --extra docs` | this site: `uv run --extra docs mkdocs serve` |

With the `julia` extra installed, always pass `--extra julia` to `uv run` and `uv sync`, or uv
removes juliacall from the environment.

## Working copies on network shares

If the working copy is on a share without symlinks, hard links, `chmod` or read-write opens (SMB
through gvfs, for instance), keep what needs them on a local disk:

```bash
git clone --separate-git-dir "$HOME/git/gridfm-datakit-distr.git" <url> gridfm-datakit-distr
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/gridfm-datakit-distr"
export PYTEST_ADDOPTS="-o cache_dir=$HOME/.cache/gridfm-datakit-distr-dev/pytest"
git config core.fileMode false
```
