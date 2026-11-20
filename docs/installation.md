# Installation

You need Python 3.10, 3.11 or 3.12. The power flow runs in OpenDSS through
[DSS-Extensions](https://dss-extensions.org/) (OpenDSSDirect.py), installed from PyPI with the
package. Networks are downloaded afterwards with `fetch`
([Networks](manual/networks.md#sources-and-the-cache)).

## From the repository

```bash
pip install "gridfm-datakit-distr @ git+https://github.com/gridfm/gridfm-datakit-distr.git@v0.2.0"
gridfm_datakit_distr version
```

`@v0.2.0` is the latest release, which writes data schema 0.5.0. This documentation describes the
development version on `main` (schema 0.6.0); to install it, leave out `@v0.2.0`.

## For developers

With [uv](https://docs.astral.sh/uv/), which creates `.venv` with the versions pinned in `uv.lock`:

```bash
git clone https://github.com/gridfm/gridfm-datakit-distr.git
cd gridfm-datakit-distr
uv sync
uv run pytest -m "not slow and not julia"
```

Tests are in `tests/` (unit) and `integrationtests/` (catalog networks, golden files,
benchmarks). Markers: `slow` (> 10 s), `network_data` (needs fetched networks; skipped otherwise),
`conformance` (catalog networks), `perf` (benchmarks, run with `-m perf`), `julia` (needs Julia,
run with `DISTR_RUN_JULIA=1`). Lint and types:
`uv run ruff check && uv run ruff format --check && uv run mypy`.

## Optional extras

| Extra | Install | For |
|---|---|---|
| `opf` | `pip install "gridfm-datakit-distr[opf] @ …"` or `uv sync --extra opf` | OPF labels (CasADi with its bundled Ipopt): [OPF labels](manual/opf.md) |
| `plots` | `pip install "gridfm-datakit-distr[plots] @ …"` | `stats --plots` |
| `croissant` | `pip install "gridfm-datakit-distr[croissant] @ …"`; for development, in its own environment: its type stubs break the project's type checking | `croissant <dataset>`: validation with mlcroissant |
| `julia` | `pip install "gridfm-datakit-distr[julia] @ …"` or `uv sync --extra julia`, then `gridfm_datakit_distr setup-julia` | Julia 1.10 and PowerModelsDistribution (about 1 GB), only for the OPF cross-check tests |
| `docs` | `uv sync --extra docs` | this site: `uv run --extra docs mkdocs serve` |

With uv, run with the extras you use (`uv run --extra opf …`). With the `julia` extra installed,
always pass `--extra julia` to `uv run` and `uv sync`, or uv removes juliacall from the
environment.

## Working copies on network shares

If the working copy is on a share without symlinks, hard links, `chmod` or read-write opens (SMB
through gvfs, for instance), keep what needs them on a local disk:

```bash
git clone --separate-git-dir "$HOME/git/gridfm-datakit-distr.git" <url> gridfm-datakit-distr
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/gridfm-datakit-distr"
export PYTEST_ADDOPTS="-o cache_dir=$HOME/.cache/gridfm-datakit-distr-dev/pytest"
git config core.fileMode false
```
