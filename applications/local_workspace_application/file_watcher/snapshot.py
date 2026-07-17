# © Artur Czarnecki. All rights reserved.

"""Allowlisted filesystem metadata snapshots and change detection (LKW.7A).

Version identity is metadata-based:
canonical path + size_bytes + modified_time_ns.
Content hashing and file-body reads are out of scope for LKW.7A.
"""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.tools.providers.filesystem.allowlist import (
    require_read_allowlist_roots,
    resolve_allowed_path,
)

from local_workspace_application.file_watcher.contracts import (
    FileChange,
    FileSnapshot,
    normalize_watch_path_key,
)


def snapshot_file(
    path: str | Path,
    *,
    allowed_roots: frozenset[str],
) -> FileSnapshot:
    """Snapshot one allowlisted regular file using size and mtime_ns only."""
    roots = require_read_allowlist_roots(allowed_roots)
    raw_candidate = Path(str(path)).expanduser()
    if not raw_candidate.is_absolute():
        raise RuntimeError("path_must_be_absolute")
    try:
        if raw_candidate.is_symlink():
            raise RuntimeError("watch_symlink_not_supported")
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError("file_snapshot_failed") from None

    try:
        resolved = resolve_allowed_path(str(path), roots)
        if not resolved.exists():
            raise RuntimeError("watch_file_not_found")
        if resolved.is_symlink():
            raise RuntimeError("watch_symlink_not_supported")
        if not resolved.is_file():
            raise RuntimeError("watch_path_not_regular_file")
        stat_result = resolved.stat()
    except RuntimeError:
        raise
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise RuntimeError("watch_file_not_found") from None
        raise RuntimeError("file_snapshot_failed") from None
    return FileSnapshot(
        path=str(resolved),
        size_bytes=stat_result.st_size,
        modified_time_ns=stat_result.st_mtime_ns,
    )


def snapshot_allowed_roots(
    allowed_roots: frozenset[str],
) -> tuple[FileSnapshot, ...]:
    """Walk allowlisted roots and return deterministic regular-file snapshots."""
    roots = require_read_allowlist_roots(allowed_roots)
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.is_absolute():
            raise RuntimeError("watch_root_not_absolute")
        if not root_path.exists():
            raise RuntimeError("watch_root_not_found")
        if not root_path.is_dir():
            raise RuntimeError("watch_root_not_directory")

    snapshots_by_key: dict[str, FileSnapshot] = {}
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            dirnames[:] = [
                name for name in dirnames if not Path(dirpath, name).is_symlink()
            ]
            for name in filenames:
                candidate = Path(dirpath, name)
                if candidate.is_symlink():
                    continue
                if not candidate.is_file():
                    continue
                try:
                    resolved = resolve_allowed_path(str(candidate), roots)
                    if resolved.is_symlink() or not resolved.is_file():
                        continue
                    stat_result = resolved.stat()
                except FileNotFoundError:
                    continue
                except RuntimeError as exc:
                    if str(exc) in {
                        "path_not_in_allowlist",
                        "path_must_be_absolute",
                        "read_allowlist_not_configured",
                    }:
                        raise
                    raise RuntimeError("file_snapshot_failed") from None
                except OSError:
                    raise RuntimeError("file_snapshot_failed") from None
                snapshot = FileSnapshot(
                    path=str(resolved),
                    size_bytes=stat_result.st_size,
                    modified_time_ns=stat_result.st_mtime_ns,
                )
                snapshots_by_key[normalize_watch_path_key(snapshot.path)] = snapshot

    ordered_keys = sorted(snapshots_by_key.keys())
    return tuple(snapshots_by_key[key] for key in ordered_keys)


def detect_file_changes(
    previous: tuple[FileSnapshot, ...],
    current: tuple[FileSnapshot, ...],
) -> tuple[FileChange, ...]:
    """Compare metadata snapshots without touching the filesystem."""
    previous_by_key = {
        normalize_watch_path_key(snapshot.path): snapshot for snapshot in previous
    }
    current_by_key = {
        normalize_watch_path_key(snapshot.path): snapshot for snapshot in current
    }
    changes: list[FileChange] = []
    for key in sorted(set(previous_by_key) | set(current_by_key)):
        prev = previous_by_key.get(key)
        curr = current_by_key.get(key)
        if prev is None and curr is not None:
            changes.append(
                FileChange(
                    kind="created",
                    path=curr.path,
                    previous=None,
                    current=curr,
                )
            )
            continue
        if prev is not None and curr is None:
            changes.append(
                FileChange(
                    kind="deleted",
                    path=prev.path,
                    previous=prev,
                    current=None,
                )
            )
            continue
        if prev is None or curr is None:
            continue
        if (
            prev.size_bytes == curr.size_bytes
            and prev.modified_time_ns == curr.modified_time_ns
        ):
            continue
        changes.append(
            FileChange(
                kind="modified",
                path=curr.path,
                previous=prev,
                current=curr,
            )
        )
    return tuple(changes)
