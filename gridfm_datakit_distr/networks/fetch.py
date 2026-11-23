"""Download catalog sources into the cache and verify every file.

A source is installed atomically: files go to a staging directory, are verified (SHA-256
against the manifest, plus listed size and git blob SHA-1 where the remote provides them),
archives are extracted, and only then the staging directory replaces the cached copy.
"""

from __future__ import annotations

import fnmatch
import hashlib
import http.client
import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gridfm_datakit_distr import __version__
from gridfm_datakit_distr.networks.cache import (
    INSTALLED,
    MARKER,
    cache_root,
    is_fetched,
    read_marker,
    source_dir,
)
from gridfm_datakit_distr.networks.catalog import (
    ArchivesSource,
    CatalogError,
    CsiroDapSource,
    GithubSource,
    SourceT,
    format_manifest,
    parse_manifest,
    safe_relative_path,
)

USER_AGENT = f"gridfm-datakit-distr/{__version__}"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_ATTEMPTS = 4

# External RAR extractors, tried in order: (executable, argument template).
_RAR_TOOLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("unrar", ("x", "-o+", "-idq", "{archive}", "{dest}/")),
    ("7z", ("x", "-y", "-o{dest}", "{archive}")),
    ("7zz", ("x", "-y", "-o{dest}", "{archive}")),
    ("bsdtar", ("-xf", "{archive}", "-C", "{dest}")),
)


class FetchError(RuntimeError):
    """A source could not be downloaded or installed."""


class ChecksumError(FetchError):
    """Downloaded or local files do not match the expected checksums or sizes."""


@dataclass(frozen=True)
class RemoteFile:
    """A file offered by a source: where it goes, where it comes from, what the remote says."""

    path: str
    url: str
    size: int | None = None
    git_blob_sha1: str | None = None


@dataclass(frozen=True)
class FetchResult:
    """Outcome of :func:`fetch_source`."""

    source_id: str
    directory: Path
    n_files: int
    skipped: bool = False
    unverified: dict[str, str] = field(default_factory=dict)
    """Files accepted without a manifest checksum (``--allow-unverified``) -> computed SHA-256."""
    lock_file: Path | None = None


def _http_get(url: str, accept: str | None = None) -> bytes:
    """GET with retries on transient failures (HTTP 429/5xx, network errors)."""
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    delay = 1.0
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_STATUS or attempt == _ATTEMPTS:
                raise FetchError(f"GET {url}: HTTP {exc.code}") from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as exc:
            if attempt == _ATTEMPTS:
                raise FetchError(f"GET {url}: {exc}") from exc
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def git_blob_sha1(data: bytes) -> str:
    """Git object id of a blob with this content."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------- listings


def _raw_github_url(source: GithubSource, rel: str) -> str:
    path = f"{source.subdir.strip('/')}/{rel}"
    return (
        f"https://raw.githubusercontent.com/{source.repo}/{source.revision}/"
        f"{urllib.parse.quote(path)}"
    )


def _list_github(source: GithubSource, discover: bool) -> list[RemoteFile]:
    if not discover:
        # The manifest is complete: raw files by path, no API call (no rate limit, no outage).
        return [RemoteFile(rel, _raw_github_url(source, rel)) for rel in sorted(source.manifest)]
    api = f"https://api.github.com/repos/{source.repo}/git/trees/{source.revision}?recursive=1"
    tree = json.loads(_http_get(api, accept="application/vnd.github+json"))
    if tree.get("truncated"):
        raise FetchError(f"{source.id}: GitHub returned a truncated tree listing")
    prefix = source.subdir.strip("/") + "/"
    files = []
    for entry in tree["tree"]:
        if entry["type"] != "blob" or not entry["path"].startswith(prefix):
            continue
        rel = entry["path"][len(prefix) :]
        if not any(
            rel == inc or (inc.endswith("/") and rel.startswith(inc)) for inc in source.include
        ):
            continue
        files.append(RemoteFile(rel, _raw_github_url(source, rel), entry.get("size"), entry["sha"]))
    return files


def _list_csiro_dap(source: CsiroDapSource) -> list[RemoteFile]:
    url = f"{source.api_base}/collections/{source.collection}v{source.version}/data"
    listing = json.loads(_http_get(url, accept="application/json"))
    return [
        RemoteFile(item["filename"], item["link"]["href"], item.get("fileSize"))
        for item in listing["file"]
        if not any(fnmatch.fnmatch(item["filename"], pat) for pat in source.exclude)
    ]


def _list_archives(source: ArchivesSource) -> list[RemoteFile]:
    return [RemoteFile(name, url) for name, url in source.archives.items()]


def list_remote(source: SourceT, discover: bool = False) -> list[RemoteFile]:
    """Files to download for ``source``.

    With ``discover`` (or an empty manifest) GitHub sources are listed through the API, which
    also yields the git blob SHA-1 of each file; otherwise the manifest is the file list.
    """
    if isinstance(source, GithubSource):
        files = _list_github(source, discover or not source.manifest)
    elif isinstance(source, CsiroDapSource):
        files = _list_csiro_dap(source)
    else:
        files = _list_archives(source)
    for item in files:
        try:
            safe_relative_path(item.path)
        except CatalogError as exc:
            raise FetchError(f"{source.id}: the remote offers an unsafe path: {exc}") from exc
        if not item.url.startswith("https://"):  # urllib would also open file:// and others
            raise FetchError(f"{source.id}: {item.path}: not an https URL: {item.url}")
    return files


# ---------------------------------------------------------------- verification


def _problems(
    path: str, data: bytes, remote: RemoteFile | None, manifest: dict[str, str]
) -> list[str]:
    problems = []
    if remote is not None and remote.size is not None and len(data) != remote.size:
        problems.append(f"{path}: {len(data)} bytes, the remote lists {remote.size}")
    blob = remote.git_blob_sha1 if remote is not None else None
    if blob is not None and git_blob_sha1(data) != blob:
        problems.append(f"{path}: git blob SHA-1 differs from the pinned commit")
    expected = manifest.get(path)
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        problems.append(f"{path}: SHA-256 differs from the manifest")
    return problems


def _raise_if(problems: list[str], source_id: str, error: type[FetchError]) -> None:
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise error(f"{source_id}: {len(problems)} problem(s):\n  {shown}{more}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installed_files(directory: Path) -> dict[str, str]:
    """SHA-256 of every file under ``directory`` (extracted ones included), by relative path."""
    return {
        p.relative_to(directory).as_posix(): _sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def verify_cached(source: SourceT, root: Path | None = None) -> None:
    """Raise :class:`ChecksumError` if a file that ``fetch`` installed for ``source`` is missing
    or changed since. A copy fetched by an earlier version, which recorded no file list, is
    checked against the manifest; a source without a fetch marker has nothing to check."""
    if read_marker(source.id, root) is None:
        return
    directory = source_dir(source.id, root)
    listed = directory / INSTALLED
    expected = (
        parse_manifest(listed.read_text(encoding="utf-8"), origin=str(listed))
        if listed.is_file()
        else source.manifest
    )
    changed = [
        rel
        for rel, digest in sorted(expected.items())
        if not (directory / rel).is_file() or _sha256(directory / rel) != digest
    ]
    if changed:
        shown = ", ".join(changed[:5]) + (f" and {len(changed) - 5} more" if changed[5:] else "")
        raise ChecksumError(
            f"{source.id}: {len(changed)} cached file(s) changed since fetch ({shown}); "
            f"restore them with `gridfm_datakit_distr fetch --force {source.id}`"
        )


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _download(
    source: SourceT, staging: Path, allow_unverified: bool, workers: int
) -> dict[str, str]:
    files = list_remote(source, discover=allow_unverified)
    listed = {item.path for item in files}
    missing = sorted(set(source.manifest) - listed)
    _raise_if(
        [f"{p}: in the manifest, not offered by the remote" for p in missing], source.id, FetchError
    )
    unverified = sorted(listed - set(source.manifest))
    if unverified and not allow_unverified:
        _raise_if(
            [
                f"{p}: no checksum in the manifest (use --allow-unverified to record it)"
                for p in unverified
            ],
            source.id,
            FetchError,
        )

    def get(item: RemoteFile) -> tuple[list[str], str]:
        data = _http_get(item.url)
        problems = _problems(item.path, data, item, source.manifest)
        if not problems:
            _write(staging / item.path, data)
        return problems, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(get, files))
    _raise_if([p for problems, _ in results for p in problems], source.id, ChecksumError)
    return {
        item.path: digest
        for item, (_, digest) in zip(files, results, strict=True)
        if item.path not in source.manifest
    }


def _install_local(source: SourceT, local: Path, staging: Path) -> None:
    if not source.manifest:
        raise FetchError(f"{source.id}: no manifest, a local copy cannot be verified")
    if local.is_file():
        if len(source.manifest) != 1:
            raise FetchError(
                f"{source.id}: --from-file needs a directory holding all {len(source.manifest)} "
                f"files of the source (e.g. {sorted(source.manifest)[:3]})"
            )
        (name,) = source.manifest
        candidates = {name: local}
    elif local.is_dir():
        candidates = {rel: local / rel for rel in source.manifest}
    else:
        raise FetchError(f"{local} does not exist")
    problems = []
    for rel, path in sorted(candidates.items()):
        if not path.is_file():
            problems.append(f"{rel}: missing in {local}")
            continue
        data = path.read_bytes()
        found = _problems(rel, data, None, source.manifest)
        problems.extend(found)
        if not found:
            _write(staging / rel, data)
    _raise_if(problems, source.id, ChecksumError)


def _extract(archive: Path, dest: Path) -> None:
    """Extract a verified archive into ``dest`` (RAR through an external tool)."""
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        # ZipFile.extractall never writes outside dest (absolute paths and '..' are dropped).
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(dest)
        return
    if suffix != ".rar":
        raise FetchError(f"unsupported archive format: {archive.name}")
    errors = []
    for tool, template in _RAR_TOOLS:
        exe = shutil.which(tool)
        if exe is None:
            continue
        args = [a.format(archive=archive, dest=dest) for a in template]
        proc = subprocess.run([exe, *args], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
        # e.g. p7zip's 7z without RAR5 support: the next tool overwrites what it left
        errors.append(f"{tool}: {proc.stderr.strip()[:500]}")
    if errors:
        raise FetchError(f"no RAR extractor could read {archive.name}: {'; '.join(errors)}")
    tools = ", ".join(tool for tool, _ in _RAR_TOOLS)
    raise FetchError(
        f"cannot extract {archive.name}: no RAR extractor on PATH (one of: {tools}); "
        "install one and fetch again"
    )


def fetch_source(
    source: SourceT,
    *,
    root: Path | None = None,
    from_path: Path | None = None,
    allow_unverified: bool = False,
    force: bool = False,
    workers: int = 8,
) -> FetchResult:
    """Download (or install from ``from_path``) and verify a source into the cache.

    With ``allow_unverified``, files missing from the manifest are accepted and their computed
    SHA-256 are written to ``<cache>/<source>.unverified.sha256`` for review; they must be
    added to the manifest before the cached copy counts as verified.
    """
    root = root if root is not None else cache_root()
    target = source_dir(source.id, root)
    if from_path is None and not (force or allow_unverified) and is_fetched(source, root):
        verify_cached(source, root)
        return FetchResult(source.id, target, len(source.manifest), skipped=True)

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{source.id}-", dir=root))
    try:
        if from_path is not None:
            _install_local(source, from_path, staging)
            unverified: dict[str, str] = {}
        else:
            unverified = _download(source, staging, allow_unverified, workers)
        if isinstance(source, ArchivesSource):
            for name in source.archives:
                _extract(staging / name, staging / Path(name).stem)
        lock_file = None
        if unverified:
            lock_file = root / f"{source.id}.unverified.sha256"
            lock_file.write_text(format_manifest(unverified), encoding="utf-8")
        marker = {
            "source": source.id,
            # A copy holding unverified files never counts as verified (see is_fetched).
            "manifest_sha256": None if unverified else source.manifest_sha256,
            "n_files": len(source.manifest) + len(unverified),
            "n_unverified": len(unverified),
            "origin": str(from_path) if from_path is not None else "download",
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "generator_version": __version__,
        }
        installed = format_manifest(_installed_files(staging))  # what verify_cached checks
        (staging / INSTALLED).write_text(installed, encoding="utf-8")
        (staging / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        if target.exists():
            trash = root / f".trash-{source.id}-{uuid.uuid4().hex}"
            target.rename(trash)
            staging.rename(target)
            shutil.rmtree(trash, ignore_errors=True)
        else:
            staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return FetchResult(
        source.id,
        target,
        len(source.manifest) + len(unverified),
        unverified=unverified,
        lock_file=lock_file,
    )
