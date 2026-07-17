# © Artur Czarnecki. All rights reserved.

"""Tests for LKW.7A incremental batch coalescing and watcher job builder."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from local_workspace_application.background_ingest.contracts import (
    background_ingest_idempotency_key,
    encode_background_ingest_job,
)
from local_workspace_application.file_watcher import (
    FileChange,
    FileSnapshot,
    build_file_watcher_ingest_job,
    build_incremental_file_change_batch,
    file_change_token,
)
from local_workspace_application.file_watcher.contracts import (
    IncrementalFileChangeBatch,
    normalize_watch_path_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_NAMES = frozenset(
    {
        "content",
        "chunks",
        "prompt",
        "document_text",
        "file_bytes",
        "embedding",
    }
)


def _snap(path: Path, *, size: int, mtime: int) -> FileSnapshot:
    return FileSnapshot(
        path=str(path),
        size_bytes=size,
        modified_time_ns=mtime,
    )


def _abs(tmp_path: Path, name: str) -> Path:
    return (tmp_path / name).resolve()


def test_same_snapshots_different_order_same_token(tmp_path: Path) -> None:
    a = _snap(_abs(tmp_path, "a.txt"), size=1, mtime=10)
    b = _snap(_abs(tmp_path, "b.txt"), size=2, mtime=20)

    assert file_change_token((a, b)) == file_change_token((b, a))


def test_duplicate_snapshots_do_not_change_token(tmp_path: Path) -> None:
    a = _snap(_abs(tmp_path, "a.txt"), size=1, mtime=10)

    assert file_change_token((a, a)) == file_change_token((a,))


def test_size_and_mtime_change_tokens(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    base = _snap(path, size=1, mtime=10)
    size_changed = _snap(path, size=2, mtime=10)
    mtime_changed = _snap(path, size=1, mtime=11)

    assert file_change_token((base,)) != file_change_token((size_changed,))
    assert file_change_token((base,)) != file_change_token((mtime_changed,))


def test_created_vs_modified_same_final_snapshot_same_token(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=3, mtime=30)
    created = FileChange(kind="created", path=final.path, current=final)
    previous = _snap(path, size=1, mtime=10)
    modified = FileChange(
        kind="modified",
        path=final.path,
        previous=previous,
        current=final,
    )

    created_batch = build_incremental_file_change_batch([created])
    modified_batch = build_incremental_file_change_batch([modified])

    assert created_batch.change_token == modified_batch.change_token
    assert created_batch.change_token == file_change_token((final,))


def test_empty_snapshots_produce_none_token() -> None:
    assert file_change_token(()) is None


def test_duplicate_events_collapse_to_one_source_snapshot(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=10)
    change = FileChange(kind="created", path=snap.path, current=snap)

    batch = build_incremental_file_change_batch([change, change, change])

    assert batch.source_snapshots == (snap,)
    assert batch.deleted_paths == ()


def test_created_then_modified_uses_final_snapshot(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    first = _snap(path, size=1, mtime=10)
    final = _snap(path, size=9, mtime=90)
    batch = build_incremental_file_change_batch(
        [
            FileChange(kind="created", path=first.path, current=first),
            FileChange(
                kind="modified",
                path=final.path,
                previous=first,
                current=final,
            ),
        ]
    )

    assert batch.source_snapshots == (final,)
    assert batch.change_token == file_change_token((final,))


def test_modified_then_deleted_becomes_deletion_only(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=10)
    current = _snap(path, size=2, mtime=20)
    batch = build_incremental_file_change_batch(
        [
            FileChange(
                kind="modified",
                path=current.path,
                previous=previous,
                current=current,
            ),
            FileChange(kind="deleted", path=previous.path, previous=previous),
        ]
    )

    assert batch.source_snapshots == ()
    assert batch.deleted_paths == (previous.path,)
    assert batch.change_token is None


def test_deleted_then_created_becomes_actionable(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=10)
    created = _snap(path, size=5, mtime=50)
    batch = build_incremental_file_change_batch(
        [
            FileChange(kind="deleted", path=previous.path, previous=previous),
            FileChange(kind="created", path=created.path, current=created),
        ]
    )

    assert batch.source_snapshots == (created,)
    assert batch.deleted_paths == ()
    assert batch.change_token is not None


def test_deleted_only_batch_has_none_token(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=10)
    batch = build_incremental_file_change_batch(
        [FileChange(kind="deleted", path=previous.path, previous=previous)]
    )

    assert batch.change_token is None
    assert batch.source_paths == ()


def test_source_and_deleted_paths_are_sorted_unique(tmp_path: Path) -> None:
    a = _snap(_abs(tmp_path, "a.txt"), size=1, mtime=1)
    b = _snap(_abs(tmp_path, "b.txt"), size=1, mtime=1)
    c_prev = _snap(_abs(tmp_path, "c.txt"), size=1, mtime=1)
    batch = build_incremental_file_change_batch(
        [
            FileChange(kind="created", path=b.path, current=b),
            FileChange(kind="created", path=a.path, current=a),
            FileChange(kind="deleted", path=c_prev.path, previous=c_prev),
            FileChange(kind="deleted", path=c_prev.path, previous=c_prev),
        ]
    )

    assert batch.source_paths == tuple(sorted(batch.source_paths))
    assert len(batch.source_paths) == len(set(batch.source_paths))
    assert batch.deleted_paths == tuple(sorted(batch.deleted_paths))
    assert len(batch.deleted_paths) == 1


def test_job_builder_mapping_and_canonical_paths(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "watched.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"payload")
    snap = FileSnapshot(
        path=str(target.resolve()),
        size_bytes=target.stat().st_size,
        modified_time_ns=target.stat().st_mtime_ns,
    )
    batch = build_incremental_file_change_batch(
        [FileChange(kind="created", path=snap.path, current=snap)]
    )
    roots = frozenset({str(tmp_path.resolve())})

    job = build_file_watcher_ingest_job(
        batch,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=roots,
        run_id="run-1",
        correlation_id="corr-1",
        priority="high",
    )

    assert job.requested_by == "lkw.file_watcher"
    assert job.reason == "lkw.7.incremental_change"
    assert job.change_token == batch.change_token
    assert job.source_paths == (str(target.resolve()),)
    assert job.run_id == "run-1"
    assert job.correlation_id == "corr-1"
    assert job.priority == "high"


def test_job_builder_rejects_deletion_only_batch(tmp_path: Path) -> None:
    previous = _snap(_abs(tmp_path, "gone.txt"), size=1, mtime=1)
    batch = build_incremental_file_change_batch(
        [FileChange(kind="deleted", path=previous.path, previous=previous)]
    )
    roots = frozenset({str(tmp_path.resolve())})

    with pytest.raises(RuntimeError, match="no_actionable_file_changes"):
        build_file_watcher_ingest_job(
            batch,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=roots,
        )


def test_job_builder_rejects_source_outside_allowlist(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outsider = tmp_path_factory.mktemp("outside") / "x.txt"
    outsider.write_bytes(b"x")
    snap = FileSnapshot(
        path=str(outsider.resolve()),
        size_bytes=1,
        modified_time_ns=1,
    )
    batch = build_incremental_file_change_batch(
        [FileChange(kind="created", path=snap.path, current=snap)]
    )

    with pytest.raises(RuntimeError, match="path_not_in_allowlist"):
        build_file_watcher_ingest_job(
            batch,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(allowed.resolve())}),
        )


def test_watcher_idempotency_same_batch_same_key_later_version_differs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "watched.txt"
    target.write_bytes(b"v1")
    snap_v1 = FileSnapshot(
        path=str(target.resolve()),
        size_bytes=target.stat().st_size,
        modified_time_ns=target.stat().st_mtime_ns,
    )
    batch_v1 = build_incremental_file_change_batch(
        [FileChange(kind="created", path=snap_v1.path, current=snap_v1)]
    )
    roots = frozenset({str(tmp_path.resolve())})
    job_a = build_file_watcher_ingest_job(
        batch_v1,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=roots,
        run_id="run-a",
    )
    job_b = build_file_watcher_ingest_job(
        batch_v1,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=roots,
        run_id="run-b",
        correlation_id="corr-b",
    )
    assert background_ingest_idempotency_key(
        job_a
    ) == background_ingest_idempotency_key(job_b)

    snap_v2 = FileSnapshot(
        path=str(target.resolve()),
        size_bytes=snap_v1.size_bytes + 1,
        modified_time_ns=snap_v1.modified_time_ns + 1,
    )
    batch_v2 = build_incremental_file_change_batch(
        [
            FileChange(
                kind="modified",
                path=snap_v2.path,
                previous=snap_v1,
                current=snap_v2,
            )
        ]
    )
    job_v2 = build_file_watcher_ingest_job(
        batch_v2,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=roots,
    )
    assert background_ingest_idempotency_key(
        job_v2
    ) != background_ingest_idempotency_key(job_a)


def test_contracts_and_job_payload_exclude_raw_content_fields(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=snap.path, current=snap)
    batch = build_incremental_file_change_batch([change])
    path.write_bytes(b"x")
    job = build_file_watcher_ingest_job(
        batch,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
    )
    payload = json.loads(encode_background_ingest_job(job).decode("utf-8"))
    contract_dump = {
        "snapshot": snap.model_dump(),
        "change": change.model_dump(),
        "batch": batch.model_dump(),
        "job": payload,
    }
    encoded = json.dumps(contract_dump)
    assert _FORBIDDEN_NAMES.isdisjoint(payload.keys())
    assert _FORBIDDEN_NAMES.isdisjoint(snap.model_dump().keys())
    assert not re.search(
        r"\b(content|chunks|prompt|document_text|file_bytes|embedding)\b",
        encoded,
    )


def test_equivalent_snapshot_paths_canonicalize_identically(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    canonical = tmp_path / "file.txt"
    lexical = tmp_path / "nested" / ".." / "file.txt"
    first = FileSnapshot(path=str(canonical), size_bytes=1, modified_time_ns=10)
    second = FileSnapshot(path=str(lexical), size_bytes=1, modified_time_ns=10)

    assert first.path == second.path
    assert normalize_watch_path_key(first.path) == normalize_watch_path_key(second.path)
    assert ".." not in Path(first.path).parts


def test_equivalent_lexical_paths_generate_same_token(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    canonical = tmp_path / "file.txt"
    lexical = tmp_path / "nested" / ".." / "file.txt"
    first = FileSnapshot(path=str(canonical), size_bytes=4, modified_time_ns=40)
    second = FileSnapshot(path=str(lexical), size_bytes=4, modified_time_ns=40)

    assert file_change_token((first,)) == file_change_token((second,))


def test_equivalent_paths_coalesce_into_one_batch_entry(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    canonical = tmp_path / "file.txt"
    lexical = tmp_path / "nested" / ".." / "file.txt"
    first = FileSnapshot(path=str(canonical), size_bytes=1, modified_time_ns=10)
    second = FileSnapshot(path=str(lexical), size_bytes=2, modified_time_ns=20)
    batch = build_incremental_file_change_batch(
        [
            FileChange(kind="created", path=first.path, current=first),
            FileChange(
                kind="modified",
                path=second.path,
                previous=first,
                current=second,
            ),
        ]
    )

    assert len(batch.source_snapshots) == 1
    assert batch.source_paths == (second.path,)
    assert batch.change_token == file_change_token((second,))


def test_canonicalized_deleted_paths_cannot_overlap_sources(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    canonical = tmp_path / "file.txt"
    lexical_deleted = tmp_path / "nested" / ".." / "file.txt"
    snap = FileSnapshot(path=str(canonical), size_bytes=1, modified_time_ns=1)

    with pytest.raises(ValueError, match="cannot exist in both"):
        IncrementalFileChangeBatch(
            source_snapshots=(snap,),
            deleted_paths=(str(lexical_deleted),),
            change_token=file_change_token((snap,)),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows path case folding only")
def test_normalize_watch_path_key_folds_case_on_windows(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    upper = str(base / "Docs" / "File.TXT")
    lower = str(base / "docs" / "file.txt")

    assert normalize_watch_path_key(upper) == normalize_watch_path_key(lower)
