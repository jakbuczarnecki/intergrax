# © Artur Czarnecki. All rights reserved.

"""Incremental change coalescing and watcher ingest job construction (LKW.7A).

LKW.7A builds validated LkwBackgroundIngestJob values only.
Message-bus enqueue and watcher runtime loops belong to LKW.7B.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from intergrax.tools.providers.filesystem.allowlist import (
    require_read_allowlist_roots,
    resolve_allowed_path,
)

from local_workspace_application.background_ingest.contracts import (
    LkwBackgroundIngestJob,
)
from local_workspace_application.file_watcher.contracts import (
    FileChange,
    FileSnapshot,
    IncrementalFileChangeBatch,
    normalize_watch_path_key,
)


def file_change_token(
    snapshots: Iterable[FileSnapshot],
) -> str | None:
    """Deterministic sha256 token for final actionable file versions."""
    by_key: dict[str, FileSnapshot] = {}
    for snapshot in snapshots:
        by_key[normalize_watch_path_key(snapshot.path)] = snapshot
    if not by_key:
        return None
    payload = [
        {
            "modified_time_ns": by_key[key].modified_time_ns,
            "path": by_key[key].path,
            "size_bytes": by_key[key].size_bytes,
        }
        for key in sorted(by_key.keys())
    ]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def build_incremental_file_change_batch(
    changes: Iterable[FileChange],
) -> IncrementalFileChangeBatch:
    """Coalesce path events to one final state; last change wins per path."""
    last_by_key: dict[str, FileChange] = {}
    for change in changes:
        last_by_key[normalize_watch_path_key(change.path)] = change

    source_by_key: dict[str, FileSnapshot] = {}
    deleted_by_key: dict[str, str] = {}
    for key in sorted(last_by_key.keys()):
        change = last_by_key[key]
        if change.kind == "deleted":
            deleted_by_key[key] = change.path
            continue
        if change.current is None:
            raise RuntimeError("file_snapshot_failed")
        source_by_key[key] = change.current

    source_snapshots = tuple(source_by_key[key] for key in sorted(source_by_key.keys()))
    deleted_paths = tuple(deleted_by_key[key] for key in sorted(deleted_by_key.keys()))
    return IncrementalFileChangeBatch(
        source_snapshots=source_snapshots,
        deleted_paths=deleted_paths,
        change_token=file_change_token(source_snapshots),
    )


def build_file_watcher_ingest_job(
    batch: IncrementalFileChangeBatch,
    *,
    tenant_id: str,
    workspace_id: str,
    collection_id: str,
    allowed_roots: frozenset[str],
    run_id: str | None = None,
    correlation_id: str | None = None,
    priority: str = "normal",
) -> LkwBackgroundIngestJob:
    """Build a background-ingest job from an actionable incremental batch."""
    roots = require_read_allowlist_roots(allowed_roots)
    if not batch.source_snapshots:
        raise RuntimeError("no_actionable_file_changes")
    if batch.change_token is None:
        raise RuntimeError("missing_change_token")

    canonical_by_key: dict[str, str] = {}
    for snapshot in batch.source_snapshots:
        resolved = resolve_allowed_path(snapshot.path, roots)
        path_text = str(resolved)
        canonical_by_key[normalize_watch_path_key(path_text)] = path_text

    source_paths = tuple(
        canonical_by_key[key] for key in sorted(canonical_by_key.keys())
    )
    return LkwBackgroundIngestJob(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        collection_id=collection_id,
        source_paths=source_paths,
        requested_by="lkw.file_watcher",
        reason="lkw.7.incremental_change",
        change_token=batch.change_token,
        run_id=run_id,
        correlation_id=correlation_id,
        priority=priority,
    )
