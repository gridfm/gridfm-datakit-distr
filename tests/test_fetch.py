"""Tests for fetching sources into the cache. Offline: HTTP is replaced by a fake."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridfm_datakit_distr.cli import app
from gridfm_datakit_distr.networks import fetch as fetch_mod
from gridfm_datakit_distr.networks.cache import is_fetched, read_marker, source_dir
from gridfm_datakit_distr.networks.catalog import ArchivesSource, CsiroDapSource, GithubSource
from gridfm_datakit_distr.networks.fetch import (
    ChecksumError,
    FetchError,
    FetchResult,
    fetch_source,
    git_blob_sha1,
    verify_cached,
)

REV = "0" * 40
FILES = {
    "a/master.dss": b"Clear\nNew Circuit.x\nRedirect lines.dss\n",
    "a/lines.dss": b"New Line.l1 bus1=a bus2=b\n",
    "common.dss": b"! shared definitions\n",
}
META = {"title": "t", "homepage": "h", "license": "l", "citation": "c"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeWeb:
    """Stands in for ``fetch._http_get``: url -> bytes, 404 otherwise."""

    def __init__(self) -> None:
        self.pages: dict[str, bytes] = {}
        self.requests: list[str] = []

    def __call__(self, url: str, accept: str | None = None) -> bytes:
        self.requests.append(url)
        if url not in self.pages:
            raise FetchError(f"GET {url}: HTTP 404")
        return self.pages[url]


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> FakeWeb:
    fake = FakeWeb()
    monkeypatch.setattr(fetch_mod, "_http_get", fake)
    return fake


def github_source(manifest: dict[str, str] | None = None) -> GithubSource:
    if manifest is None:
        manifest = {path: sha256(data) for path, data in FILES.items()}
    return GithubSource(
        id="gh", kind="github", repo="o/r", revision=REV, subdir="sub",
        include=["a/", "common.dss"], manifest=manifest, **META,
    )  # fmt: skip


def serve_github(
    web: FakeWeb, served: dict[str, bytes], listed: dict[str, bytes] | None = None
) -> None:
    """Publish a tree listing (blob ids from ``listed``) and raw files (``served``)."""
    listed = served if listed is None else listed
    tree = [
        {"path": f"sub/{path}", "type": "blob", "sha": git_blob_sha1(data), "size": len(data)}
        for path, data in listed.items()
    ]
    tree += [
        {"path": "sub/a", "type": "tree", "sha": "t"},
        {"path": "sub/other/not_included.dss", "type": "blob", "sha": "s", "size": 1},
    ]
    web.pages[f"https://api.github.com/repos/o/r/git/trees/{REV}?recursive=1"] = json.dumps(
        {"tree": tree, "truncated": False}
    ).encode()
    for path, data in served.items():
        web.pages[f"https://raw.githubusercontent.com/o/r/{REV}/sub/{path}"] = data


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.name.startswith((".staging", ".trash")))


# ---------------------------------------------------------------- downloads


def test_github_source_is_downloaded_and_verified(tmp_path: Path, web: FakeWeb) -> None:
    serve_github(web, FILES)
    source = github_source()
    result = fetch_source(source, root=tmp_path)

    assert result.n_files == 3
    assert not result.skipped
    for path, data in FILES.items():
        assert (tmp_path / "gh" / path).read_bytes() == data
    assert not (tmp_path / "gh" / "other").exists()
    assert is_fetched(source, tmp_path)
    marker = read_marker("gh", tmp_path)
    assert marker is not None
    assert marker["origin"] == "download"
    assert leftovers(tmp_path) == []
    # A complete manifest is the file list: the GitHub API is not needed.
    assert not any(url.startswith("https://api.github.com") for url in web.requests)


def test_fetched_source_is_not_downloaded_again(tmp_path: Path, web: FakeWeb) -> None:
    serve_github(web, FILES)
    source = github_source()
    fetch_source(source, root=tmp_path)
    n_requests = len(web.requests)
    assert fetch_source(source, root=tmp_path).skipped
    assert len(web.requests) == n_requests


def test_a_cached_file_changed_after_fetch_is_detected(tmp_path: Path, web: FakeWeb) -> None:
    serve_github(web, FILES)
    source = github_source()
    fetch_source(source, root=tmp_path)
    (tmp_path / "gh" / "a/lines.dss").write_bytes(b"New Line.l1 bus1=a bus2=c\n")
    (tmp_path / "gh" / "common.dss").unlink()
    with pytest.raises(ChecksumError, match=r"2 cached file\(s\) .* \(a/lines\.dss, common\.dss\)"):
        fetch_source(source, root=tmp_path)
    assert is_fetched(source, tmp_path)  # the marker alone still says fetched
    fetch_source(source, root=tmp_path, force=True)  # --force restores it
    assert fetch_source(source, root=tmp_path).skipped


def test_a_copy_fetched_before_the_file_list_is_checked_against_the_manifest(
    tmp_path: Path, web: FakeWeb
) -> None:
    serve_github(web, FILES)
    source = github_source()
    fetch_source(source, root=tmp_path)
    (tmp_path / "gh" / ".fetch.sha256").unlink()
    verify_cached(source, tmp_path)
    (tmp_path / "gh" / "common.dss").write_bytes(b"changed\n")
    with pytest.raises(ChecksumError, match=r"common\.dss"):
        verify_cached(source, tmp_path)


def test_corrupted_download_fails_and_installs_nothing(tmp_path: Path, web: FakeWeb) -> None:
    corrupted = {**FILES, "a/lines.dss": b"New Line.l1 bus1=a bus2=XX\n"}
    serve_github(
        web, corrupted
    )  # listing consistent with the corrupted bytes: SHA-256 must catch it
    with pytest.raises(ChecksumError, match=r"a/lines\.dss: SHA-256 differs"):
        fetch_source(github_source(), root=tmp_path)
    assert not source_dir("gh", tmp_path).exists()
    assert leftovers(tmp_path) == []


def test_discovery_checks_files_against_the_pinned_commit(tmp_path: Path, web: FakeWeb) -> None:
    corrupted = {**FILES, "common.dss": b"tampered\n"}
    serve_github(web, corrupted, listed=FILES)
    with pytest.raises(ChecksumError, match="git blob SHA-1"):
        fetch_source(github_source(manifest={}), root=tmp_path, allow_unverified=True)


def test_failed_refetch_keeps_the_verified_copy(tmp_path: Path, web: FakeWeb) -> None:
    serve_github(web, FILES)
    source = github_source()
    fetch_source(source, root=tmp_path)
    serve_github(web, {**FILES, "common.dss": b"tampered\n"})
    with pytest.raises(ChecksumError):
        fetch_source(source, root=tmp_path, force=True)
    assert (tmp_path / "gh" / "common.dss").read_bytes() == FILES["common.dss"]
    assert is_fetched(source, tmp_path)


def test_manifest_file_missing_from_remote_fails(tmp_path: Path, web: FakeWeb) -> None:
    serve_github(web, {k: v for k, v in FILES.items() if k != "common.dss"})
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_source(github_source(), root=tmp_path)
    with pytest.raises(FetchError, match="not offered by the remote"):
        fetch_source(github_source(), root=tmp_path, allow_unverified=True)


def test_discovered_files_are_recorded_for_review(tmp_path: Path, web: FakeWeb) -> None:
    serve_github(web, FILES)
    partial = {p: sha256(d) for p, d in FILES.items() if p != "a/lines.dss"}
    fetch_source(github_source(partial), root=tmp_path)  # the manifest is the file list
    assert not (tmp_path / "gh" / "a/lines.dss").exists()

    result = fetch_source(github_source(partial), root=tmp_path, allow_unverified=True)
    assert result.unverified == {"a/lines.dss": sha256(FILES["a/lines.dss"])}
    assert result.lock_file is not None
    assert result.lock_file.read_text() == f"{sha256(FILES['a/lines.dss'])}  a/lines.dss\n"
    assert not is_fetched(github_source(partial), tmp_path)  # unverified copy never counts


def test_csiro_listing_sizes_and_excludes(tmp_path: Path, web: FakeWeb) -> None:
    data = {"F/n_1/Feeder_1/Master.dss": b"Clear\n", "README.md": b"# readme\n"}
    listing = [
        {"filename": path, "fileSize": len(blob), "link": {"href": f"https://s3.test/{path}"}}
        for path, blob in data.items()
    ] + [{"filename": "F/.DS_Store", "fileSize": 4, "link": {"href": "https://s3.test/ds"}}]
    web.pages["https://dap.test/ws/collections/7v1/data"] = json.dumps({"file": listing}).encode()
    for path, blob in data.items():
        web.pages[f"https://s3.test/{path}"] = blob
    source = CsiroDapSource(
        id="dap", kind="csiro_dap", collection=7, version=1, api_base="https://dap.test/ws",
        exclude=["*.DS_Store"], manifest={p: sha256(b) for p, b in data.items()}, **META,
    )  # fmt: skip
    fetch_source(source, root=tmp_path)
    assert (tmp_path / "dap" / "F/n_1/Feeder_1/Master.dss").read_bytes() == b"Clear\n"
    assert "https://s3.test/ds" not in web.requests

    web.pages["https://s3.test/README.md"] = b"# readme, now longer\n"
    with pytest.raises(ChecksumError, match="the remote lists"):
        fetch_source(source, root=tmp_path, force=True)

    without_readme = source.model_copy(
        update={"manifest": {"F/n_1/Feeder_1/Master.dss": sha256(b"Clear\n")}}
    )
    with pytest.raises(FetchError, match=r"README\.md: no checksum in the manifest"):
        fetch_source(without_readme, root=tmp_path, force=True)


def test_unsafe_remote_path_is_rejected(tmp_path: Path, web: FakeWeb) -> None:
    listing = [{"filename": "../escape.dss", "fileSize": 1, "link": {"href": "https://s3.test/x"}}]
    web.pages["https://dap.test/ws/collections/7v1/data"] = json.dumps({"file": listing}).encode()
    source = CsiroDapSource(
        id="dap", kind="csiro_dap", collection=7, version=1, api_base="https://dap.test/ws", **META
    )
    with pytest.raises(FetchError, match="unsafe path"):
        fetch_source(source, root=tmp_path, allow_unverified=True)


# ---------------------------------------------------------------- local copies


def test_from_file_directory_is_verified_and_installed(tmp_path: Path, web: FakeWeb) -> None:
    local = tmp_path / "manual"
    for path, data in FILES.items():
        (local / path).parent.mkdir(parents=True, exist_ok=True)
        (local / path).write_bytes(data)
    cache = tmp_path / "cache"
    source = github_source()
    fetch_source(source, root=cache, from_path=local)
    assert web.requests == []
    assert is_fetched(source, cache)
    marker = read_marker("gh", cache)
    assert marker is not None
    assert marker["origin"] == str(local)


def test_from_file_detects_corrupted_and_missing_files(tmp_path: Path) -> None:
    local = tmp_path / "manual"
    for path, data in FILES.items():
        (local / path).parent.mkdir(parents=True, exist_ok=True)
        (local / path).write_bytes(data)
    (local / "a/lines.dss").write_bytes(b"corrupted\n")
    (local / "common.dss").unlink()
    with pytest.raises(ChecksumError) as info:
        fetch_source(github_source(), root=tmp_path / "cache", from_path=local)
    assert "a/lines.dss: SHA-256 differs" in str(info.value)
    assert "common.dss: missing" in str(info.value)
    assert not (tmp_path / "cache" / "gh").exists()


# ---------------------------------------------------------------- archives


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def archives_source(name: str, blob: bytes) -> ArchivesSource:
    return ArchivesSource(
        id="arc", kind="archives", archives={name: f"https://files.test/{name}"},
        manifest={name: sha256(blob)}, **META,
    )  # fmt: skip


def test_zip_archive_is_verified_and_extracted(tmp_path: Path, web: FakeWeb) -> None:
    blob = make_zip({"Master/master.dss": b"Clear\n"})
    web.pages["https://files.test/Pack.zip"] = blob
    source = archives_source("Pack.zip", blob)
    fetch_source(source, root=tmp_path)
    master = tmp_path / "arc" / "Pack" / "Master" / "master.dss"
    assert master.read_bytes() == b"Clear\n"
    # extracted files are recorded at install, so a change to them is caught too
    master.write_bytes(b"Clear\nNew Load.extra\n")
    with pytest.raises(ChecksumError, match=r"Pack/Master/master\.dss"):
        verify_cached(source, tmp_path)


def test_a_cut_download_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

    calls: list[str] = []

    def urlopen(request: object, timeout: float) -> Response:
        calls.append("x")
        if len(calls) == 1:  # the connection drops in the middle of the body
            raise http.client.IncompleteRead(b"partial", 100)
        return Response(b"whole")

    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    assert fetch_mod._http_get("https://files.test/x") == b"whole"
    assert len(calls) == 2


def test_rar_without_extractor_fails_with_instructions(
    tmp_path: Path, web: FakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = b"Rar!\x1a\x07\x01\x00 not really"
    web.pages["https://files.test/Rural.rar"] = blob
    monkeypatch.setattr(fetch_mod.shutil, "which", lambda tool: None)
    with pytest.raises(FetchError, match="no RAR extractor on PATH"):
        fetch_source(archives_source("Rural.rar", blob), root=tmp_path)
    assert not (tmp_path / "arc").exists()


def test_rar_goes_through_the_first_available_tool(
    tmp_path: Path, web: FakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = b"Rar!\x1a\x07\x01\x00 not really"
    web.pages["https://files.test/Rural.rar"] = blob
    fake_unrar = tmp_path / "bin" / "unrar"
    fake_unrar.parent.mkdir()
    # unrar x -o+ -idq <archive> <dest>/  ->  $5 is the destination
    fake_unrar.write_text('#!/bin/sh\nmkdir -p "$5" && echo extracted > "$5/master.dss"\n')
    fake_unrar.chmod(0o755)
    real_which = shutil.which
    monkeypatch.setattr(
        fetch_mod.shutil, "which", lambda tool: str(fake_unrar) if tool == "unrar" else None
    )
    fetch_source(archives_source("Rural.rar", blob), root=tmp_path / "cache")
    monkeypatch.setattr(fetch_mod.shutil, "which", real_which)
    assert (tmp_path / "cache" / "arc" / "Rural" / "master.dss").read_text() == "extracted\n"


def test_rar_falls_through_a_tool_that_cannot_read_it(
    tmp_path: Path, web: FakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """p7zip's ``7z`` often lacks RAR5: the next tool on the list gets its turn."""
    blob = b"Rar!\x1a\x07\x01\x00 not really"
    web.pages["https://files.test/Rural.rar"] = blob
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tools = {
        "7z": "#!/bin/sh\necho 'Can not open the file as archive' >&2\nexit 2\n",
        # 7zz x -y -o<dest> <archive>  ->  $3 is -o<dest>
        "7zz": '#!/bin/sh\nd="${3#-o}"\nmkdir -p "$d" && echo extracted > "$d/master.dss"\n',
    }
    for name, script in tools.items():
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    real_which = shutil.which
    monkeypatch.setattr(
        fetch_mod.shutil, "which", lambda tool: str(bin_dir / tool) if tool in tools else None
    )
    fetch_source(archives_source("Rural.rar", blob), root=tmp_path / "cache")
    assert (tmp_path / "cache" / "arc" / "Rural" / "master.dss").read_text() == "extracted\n"
    monkeypatch.setattr(
        fetch_mod.shutil, "which", lambda tool: str(bin_dir / tool) if tool == "7z" else None
    )
    with pytest.raises(FetchError, match="7z: Can not open the file as archive"):
        fetch_source(archives_source("Rural.rar", blob), root=tmp_path / "other")
    monkeypatch.setattr(fetch_mod.shutil, "which", real_which)


# ---------------------------------------------------------------- CLI


def test_cli_rejects_unknown_ids() -> None:
    result = CliRunner().invoke(app, ["fetch", "no_such_network"])
    assert result.exit_code == 2
    assert "neither a source nor a network" in result.output


def test_cli_from_file_needs_one_source(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["fetch", "ieee13", "csiro_enwl", "--from-file", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "exactly one source" in result.output


def test_cli_reports_each_source_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_fetch(source: object, **kwargs: object) -> FetchResult:
        calls.append(source.id)  # type: ignore[attr-defined]
        return FetchResult(source.id, tmp_path / source.id, 3)  # type: ignore[attr-defined]

    monkeypatch.setattr(fetch_mod, "fetch_source", fake_fetch)
    result = CliRunner().invoke(app, ["fetch", "ieee13", "ieee34", "--cache", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls == ["epri_ieee"]
    assert "epri_ieee: 3 files installed" in result.output
