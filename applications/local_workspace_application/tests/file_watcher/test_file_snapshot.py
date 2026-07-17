# © Artur Czarnecki. All rights reserved.

"""Tests for LKW.7A filesystem metadata snapshots and change detection."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from local_workspace_application.file_watcher import (
    detect_file_changes,
    snapshot_allowed_roots,
    snapshot_file,
)
from local_workspace_application.file_watcher.contracts import FileSnapshot

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_snapshot_file_returns_canonical_size_and_mtime(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "a.txt"
    _write(target, b"hello-world")
    roots = frozenset({str(tmp_path.resolve())})

    snapshot = snapshot_file(str(target), allowed_roots=roots)
    stat_result = target.resolve().stat()

    assert Path(snapshot.path).is_absolute()
    assert snapshot.path == str(target.resolve())
    assert snapshot.size_bytes == len(b"hello-world")
    assert snapshot.size_bytes == stat_result.st_size
    assert snapshot.modified_time_ns == stat_result.st_mtime_ns


def test_snapshot_file_canonicalizes_lexical_path_segments(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "a.txt"
    _write(target, b"hello-world")
    (tmp_path / "nested").mkdir()
    lexical = tmp_path / "nested" / ".." / "docs" / "a.txt"
    roots = frozenset({str(tmp_path.resolve())})

    snapshot = snapshot_file(str(lexical), allowed_roots=roots)

    assert snapshot.path == str(target.resolve())
    assert ".." not in Path(snapshot.path).parts


def test_snapshot_file_rejects_relative_path(tmp_path: Path) -> None:
    roots = frozenset({str(tmp_path.resolve())})
    with pytest.raises(RuntimeError, match="path_must_be_absolute"):
        snapshot_file("relative.txt", allowed_roots=roots)


def test_snapshot_file_rejects_outside_allowlist(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outsider = tmp_path_factory.mktemp("outside") / "secret.txt"
    _write(outsider, b"nope")
    roots = frozenset({str(allowed.resolve())})

    with pytest.raises(RuntimeError, match="path_not_in_allowlist"):
        snapshot_file(str(outsider), allowed_roots=roots)


def test_snapshot_file_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    roots = frozenset({str(tmp_path.resolve())})

    with pytest.raises(RuntimeError, match="watch_file_not_found"):
        snapshot_file(str(missing), allowed_roots=roots)


def test_snapshot_file_rejects_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    roots = frozenset({str(tmp_path.resolve())})

    with pytest.raises(RuntimeError, match="watch_path_not_regular_file"):
        snapshot_file(str(nested), allowed_roots=roots)


def test_snapshot_file_rejects_empty_allowlist(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    _write(target, b"x")
    with pytest.raises(RuntimeError, match="read_allowlist_not_configured"):
        snapshot_file(str(target), allowed_roots=frozenset())


def test_snapshot_allowed_roots_walks_nested_and_sorts(tmp_path: Path) -> None:
    _write(tmp_path / "z.txt", b"z")
    _write(tmp_path / "nested" / "a.txt", b"a")
    _write(tmp_path / "nested" / "b.txt", b"bb")
    (tmp_path / "nested" / "subdir").mkdir()
    roots = frozenset({str(tmp_path.resolve())})

    snapshots = snapshot_allowed_roots(roots)

    assert [Path(item.path).name for item in snapshots] == ["a.txt", "b.txt", "z.txt"]
    assert all(isinstance(item, FileSnapshot) for item in snapshots)
    assert all(item.size_bytes > 0 for item in snapshots)
    paths = [item.path for item in snapshots]
    assert paths == sorted(paths)


def test_snapshot_allowed_roots_deduplicates_overlapping_roots(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    _write(nested / "only.txt", b"one")
    roots = frozenset({str(tmp_path.resolve()), str(nested.resolve())})

    snapshots = snapshot_allowed_roots(roots)

    assert len(snapshots) == 1
    assert Path(snapshots[0].path).name == "only.txt"


def test_snapshot_allowed_roots_excludes_directories(tmp_path: Path) -> None:
    (tmp_path / "empty_dir").mkdir()
    _write(tmp_path / "file.txt", b"data")
    roots = frozenset({str(tmp_path.resolve())})

    snapshots = snapshot_allowed_roots(roots)

    assert len(snapshots) == 1
    assert Path(snapshots[0].path).name == "file.txt"


def test_detect_file_changes_created_modified_deleted_and_sorted(
    tmp_path: Path,
) -> None:
    roots = frozenset({str(tmp_path.resolve())})
    keep = tmp_path / "keep.txt"
    mutate = tmp_path / "mutate.txt"
    remove = tmp_path / "remove.txt"
    _write(keep, b"same")
    _write(mutate, b"old")
    _write(remove, b"gone")
    previous = snapshot_allowed_roots(roots)

    time.sleep(0.02)
    _write(mutate, b"new-content")
    remove.unlink()
    created = tmp_path / "created.txt"
    _write(created, b"fresh")
    # Force mtime-only change on keep if size matches across platforms
    os.utime(keep, ns=(keep.stat().st_atime_ns, keep.stat().st_mtime_ns + 1_000_000))

    current = snapshot_allowed_roots(roots)
    changes = detect_file_changes(previous, current)

    kinds = {Path(change.path).name: change.kind for change in changes}
    assert kinds["created.txt"] == "created"
    assert kinds["mutate.txt"] == "modified"
    assert kinds["remove.txt"] == "deleted"
    assert kinds["keep.txt"] == "modified"
    assert [change.path for change in changes] == sorted(
        change.path for change in changes
    )
    assert len(changes) == len({change.path for change in changes})


def test_unchanged_snapshots_produce_no_changes(tmp_path: Path) -> None:
    _write(tmp_path / "stable.txt", b"stable")
    roots = frozenset({str(tmp_path.resolve())})
    snapshots = snapshot_allowed_roots(roots)

    assert detect_file_changes(snapshots, snapshots) == ()


def test_size_change_produces_modified(tmp_path: Path) -> None:
    target = tmp_path / "sized.txt"
    _write(target, b"a")
    previous = (
        FileSnapshot(
            path=str(target.resolve()),
            size_bytes=1,
            modified_time_ns=100,
        ),
    )
    current = (
        FileSnapshot(
            path=str(target.resolve()),
            size_bytes=2,
            modified_time_ns=100,
        ),
    )

    changes = detect_file_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].kind == "modified"


def test_mtime_change_produces_modified(tmp_path: Path) -> None:
    path = str((tmp_path / "mtime.txt").resolve())
    previous = (FileSnapshot(path=path, size_bytes=1, modified_time_ns=10),)
    current = (FileSnapshot(path=path, size_bytes=1, modified_time_ns=20),)

    changes = detect_file_changes(previous, current)

    assert len(changes) == 1
    assert changes[0].kind == "modified"


def test_snapshot_file_rejects_direct_symlink_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (tmp_path / "maybe_link.txt").resolve()
    roots = frozenset({str(tmp_path.resolve())})
    resolve_called = False
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(self: Path) -> bool:
        if Path(self).expanduser() == Path(str(candidate)).expanduser():
            return True
        return original_is_symlink(self)

    def fail_if_called(*_args: object, **_kwargs: object) -> Path:
        nonlocal resolve_called
        resolve_called = True
        raise AssertionError("resolve_allowed_path must not be called")

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(
        "local_workspace_application.file_watcher.snapshot.resolve_allowed_path",
        fail_if_called,
    )

    with pytest.raises(RuntimeError, match="watch_symlink_not_supported"):
        snapshot_file(str(candidate), allowed_roots=roots)
    assert resolve_called is False


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="symlink API unavailable",
)
def test_snapshot_file_rejects_symlink_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "real.txt"
    link = tmp_path / "link.txt"
    _write(target, b"data")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation requires elevated privileges")
    roots = frozenset({str(tmp_path.resolve())})

    with pytest.raises(RuntimeError, match="watch_symlink_not_supported"):
        snapshot_file(str(link), allowed_roots=roots)
