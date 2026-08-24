# © Artur Czarnecki. All rights reserved.

"""Tests for LKW.7B2A durable file-watcher checkpoint and restore recovery."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_workspace_application.background_ingest.contracts import (
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
)
from local_workspace_application.file_watcher import (
    LKW_FILE_WATCHER_CHECKPOINT_SCHEMA_VERSION,
    FileChange,
    FileSnapshot,
    FileWatcherCheckpoint,
    FileWatcherRuntime,
    FileWatcherRuntimeConfig,
    JsonFileWatcherCheckpointStore,
    build_file_watcher_checkpoint,
    build_file_watcher_ingest_job,
    build_incremental_file_change_batch,
    decode_file_watcher_checkpoint,
    encode_file_watcher_checkpoint,
    file_change_token,
    file_watcher_checkpoint_path,
    restore_file_watcher_runtime,
)
from local_workspace_application.file_watcher import checkpoint as checkpoint_module
from local_workspace_application.file_watcher import runtime as runtime_module
from local_workspace_application.file_watcher.contracts import normalize_watch_path_key
from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueOutput

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _snap(path: Path, *, size: int, mtime: int) -> FileSnapshot:
    return FileSnapshot(path=str(path), size_bytes=size, modified_time_ns=mtime)


def _abs(tmp_path: Path, name: str) -> Path:
    return (tmp_path / name).resolve()


def _config(tmp_path: Path, **overrides: object) -> FileWatcherRuntimeConfig:
    payload: dict[str, object] = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": frozenset({str(tmp_path.resolve())}),
        "debounce_seconds": 1.0,
        "max_batch_wait_seconds": 10.0,
    }
    payload.update(overrides)
    return FileWatcherRuntimeConfig.model_validate(payload)


class _FakeSnapshotProvider:
    def __init__(
        self,
        sequence: list[tuple[FileSnapshot, ...] | BaseException],
    ) -> None:
        self._sequence = list(sequence)
        self.calls: list[frozenset[str]] = []
        self._index = 0

    def __call__(self, allowed_roots: frozenset[str]) -> tuple[FileSnapshot, ...]:
        self.calls.append(allowed_roots)
        if self._index >= len(self._sequence):
            raise AssertionError("unexpected snapshot_provider call")
        item = self._sequence[self._index]
        self._index += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeEnqueuer:
    def __init__(self, *, fail_times: int = 0) -> None:
        self.jobs: list[object] = []
        self._fail_times = fail_times
        self._calls = 0

    def __call__(self, job: object) -> MessageBusEnqueueOutput:
        self.jobs.append(job)
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("enqueue boom")
        assert isinstance(job, LkwBackgroundIngestJob)
        key = background_ingest_idempotency_key(job)
        return MessageBusEnqueueOutput(
            task_id=key,
            provider="fake",
            tenant_id=job.tenant_id,
        )


def _runtime(
    tmp_path: Path,
    *,
    snapshots: list[tuple[FileSnapshot, ...] | BaseException],
    enqueuer: _FakeEnqueuer | None = None,
    **config_overrides: object,
) -> tuple[FileWatcherRuntime, _FakeSnapshotProvider, _FakeEnqueuer]:
    provider = _FakeSnapshotProvider(snapshots)
    queue = enqueuer or _FakeEnqueuer()
    runtime = FileWatcherRuntime(
        config=_config(tmp_path, **config_overrides),
        snapshot_provider=provider,
        enqueuer=queue,
    )
    return runtime, provider, queue


def _empty_checkpoint(tmp_path: Path, **overrides: object) -> FileWatcherCheckpoint:
    payload: dict[str, object] = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": frozenset({str(tmp_path.resolve())}),
        "baseline_snapshots": (),
        "pending_changes": (),
    }
    payload.update(overrides)
    return build_file_watcher_checkpoint(**payload)  # type: ignore[arg-type]


# --- Contract tests ---


def test_valid_empty_pending_checkpoint(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    baseline = (_snap(path, size=1, mtime=1),)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=baseline,
        pending_changes=(),
    )
    assert checkpoint.pending_changes == ()
    assert checkpoint.baseline_snapshots == baseline
    assert checkpoint.schema_version == LKW_FILE_WATCHER_CHECKPOINT_SCHEMA_VERSION


def test_valid_actionable_pending_checkpoint(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    current = _snap(path, size=2, mtime=2)
    previous = _snap(path, size=1, mtime=1)
    change = FileChange(
        kind="modified",
        path=str(path),
        previous=previous,
        current=current,
    )
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(current,),
        pending_changes=(change,),
    )
    assert len(checkpoint.pending_changes) == 1
    assert checkpoint.pending_changes[0].kind == "modified"


def test_valid_deleted_pending_checkpoint(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    change = FileChange(kind="deleted", path=str(path), previous=previous)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(),
        pending_changes=(change,),
    )
    assert checkpoint.pending_changes[0].kind == "deleted"
    assert checkpoint.baseline_snapshots == ()


def test_schema_version_fixed(tmp_path: Path) -> None:
    checkpoint = _empty_checkpoint(tmp_path)
    assert checkpoint.schema_version == "lkw.file_watcher_checkpoint.v1"


def test_extra_fields_rejected(tmp_path: Path) -> None:
    root = str(tmp_path.resolve())
    with pytest.raises(ValidationError):
        FileWatcherCheckpoint.model_validate(
            {
                "tenant_id": "tenant-a",
                "workspace_id": "workspace-a",
                "collection_id": "collection-a",
                "allowed_roots": [root],
                "extra_field": "nope",
            }
        )


@pytest.mark.parametrize("field", ["tenant_id", "workspace_id", "collection_id"])
def test_blank_identity_rejected(tmp_path: Path, field: str) -> None:
    kwargs = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": frozenset({str(tmp_path.resolve())}),
        "baseline_snapshots": (),
        "pending_changes": (),
    }
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        build_file_watcher_checkpoint(**kwargs)  # type: ignore[arg-type]


def test_relative_root_rejected(tmp_path: Path) -> None:
    with pytest.raises((ValidationError, ValueError), match="absolute"):
        build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({"relative/root"}),
            baseline_snapshots=(),
            pending_changes=(),
        )


def test_duplicate_roots_rejected(tmp_path: Path) -> None:
    root = str(tmp_path.resolve())
    with pytest.raises(ValidationError, match="unique"):
        FileWatcherCheckpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=(root, root),
        )


def test_unsorted_roots_rejected(tmp_path: Path) -> None:
    a = (tmp_path / "z").resolve()
    b = (tmp_path / "a").resolve()
    a.mkdir()
    b.mkdir()
    root_a = str(a)
    root_b = str(b)
    keys = [normalize_watch_path_key(root_a), normalize_watch_path_key(root_b)]
    if keys[0] < keys[1]:
        unsorted = (root_b, root_a)
    else:
        unsorted = (root_a, root_b)
    with pytest.raises(ValidationError, match="sorted"):
        FileWatcherCheckpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=unsorted,
        )


def test_duplicate_baseline_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    with pytest.raises(ValidationError, match="unique"):
        FileWatcherCheckpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=(str(tmp_path.resolve()),),
            baseline_snapshots=(snap, snap),
        )


def test_unsorted_baseline_rejected(tmp_path: Path) -> None:
    path_a = _abs(tmp_path, "a.txt")
    path_z = _abs(tmp_path, "z.txt")
    snap_a = _snap(path_a, size=1, mtime=1)
    snap_z = _snap(path_z, size=1, mtime=1)
    key_a = normalize_watch_path_key(str(path_a))
    key_z = normalize_watch_path_key(str(path_z))
    if key_a < key_z:
        unsorted = (snap_z, snap_a)
    else:
        unsorted = (snap_a, snap_z)
    with pytest.raises(ValidationError, match="sorted"):
        FileWatcherCheckpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=(str(tmp_path.resolve()),),
            baseline_snapshots=unsorted,
        )


def test_duplicate_pending_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    change = FileChange(kind="deleted", path=str(path), previous=previous)
    with pytest.raises(ValidationError, match="unique"):
        FileWatcherCheckpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=(str(tmp_path.resolve()),),
            pending_changes=(change, change),
        )


def test_unsorted_pending_rejected(tmp_path: Path) -> None:
    path_a = _abs(tmp_path, "a.txt")
    path_z = _abs(tmp_path, "z.txt")
    change_a = FileChange(
        kind="deleted",
        path=str(path_a),
        previous=_snap(path_a, size=1, mtime=1),
    )
    change_z = FileChange(
        kind="deleted",
        path=str(path_z),
        previous=_snap(path_z, size=1, mtime=1),
    )
    key_a = normalize_watch_path_key(str(path_a))
    key_z = normalize_watch_path_key(str(path_z))
    if key_a < key_z:
        unsorted = (change_z, change_a)
    else:
        unsorted = (change_a, change_z)
    with pytest.raises(ValidationError, match="sorted"):
        FileWatcherCheckpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=(str(tmp_path.resolve()),),
            pending_changes=unsorted,
        )


# --- Baseline/pending consistency ---


def test_created_pending_missing_from_baseline_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    current = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=current)
    with pytest.raises(ValidationError, match="baseline"):
        build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(tmp_path.resolve())}),
            baseline_snapshots=(),
            pending_changes=(change,),
        )


def test_created_current_different_from_baseline_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    current = _snap(path, size=2, mtime=2)
    baseline = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=current)
    with pytest.raises(ValidationError, match="match baseline"):
        build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(tmp_path.resolve())}),
            baseline_snapshots=(baseline,),
            pending_changes=(change,),
        )


def test_modified_pending_missing_from_baseline_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    current = _snap(path, size=2, mtime=2)
    change = FileChange(
        kind="modified",
        path=str(path),
        previous=previous,
        current=current,
    )
    with pytest.raises(ValidationError, match="baseline"):
        build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(tmp_path.resolve())}),
            baseline_snapshots=(),
            pending_changes=(change,),
        )


def test_modified_current_different_from_baseline_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    current = _snap(path, size=2, mtime=2)
    baseline = _snap(path, size=3, mtime=3)
    change = FileChange(
        kind="modified",
        path=str(path),
        previous=previous,
        current=current,
    )
    with pytest.raises(ValidationError, match="match baseline"):
        build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(tmp_path.resolve())}),
            baseline_snapshots=(baseline,),
            pending_changes=(change,),
        )


def test_deleted_pending_still_in_baseline_rejected(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    change = FileChange(kind="deleted", path=str(path), previous=previous)
    with pytest.raises(ValidationError, match="absent"):
        build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(tmp_path.resolve())}),
            baseline_snapshots=(previous,),
            pending_changes=(change,),
        )


def test_created_pending_matching_baseline_accepted(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    current = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=current)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(current,),
        pending_changes=(change,),
    )
    assert checkpoint.pending_changes[0].kind == "created"


def test_modified_pending_matching_baseline_accepted(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    current = _snap(path, size=2, mtime=2)
    change = FileChange(
        kind="modified",
        path=str(path),
        previous=previous,
        current=current,
    )
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(current,),
        pending_changes=(change,),
    )
    assert checkpoint.pending_changes[0].kind == "modified"


def test_deleted_pending_absent_from_baseline_accepted(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    change = FileChange(kind="deleted", path=str(path), previous=previous)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(),
        pending_changes=(change,),
    )
    assert checkpoint.pending_changes[0].kind == "deleted"


# --- Encoding ---


def test_equivalent_checkpoints_encode_identically(tmp_path: Path) -> None:
    a = _empty_checkpoint(tmp_path)
    b = _empty_checkpoint(tmp_path)
    assert encode_file_watcher_checkpoint(a) == encode_file_watcher_checkpoint(b)


def test_encoded_json_ends_with_one_newline(tmp_path: Path) -> None:
    encoded = encode_file_watcher_checkpoint(_empty_checkpoint(tmp_path))
    assert encoded.endswith(b"\n")
    assert not encoded.endswith(b"\n\n")


def test_decode_encode_roundtrip(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    current = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=current)
    original = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(current,),
        pending_changes=(change,),
    )
    assert decode_file_watcher_checkpoint(encode_file_watcher_checkpoint(original)) == (
        original
    )


def test_encoding_has_no_indentation(tmp_path: Path) -> None:
    encoded = encode_file_watcher_checkpoint(_empty_checkpoint(tmp_path)).decode(
        "utf-8"
    )
    assert "\n " not in encoded
    assert "\n\t" not in encoded


def test_encoding_omits_timing_and_task_fields(tmp_path: Path) -> None:
    encoded = encode_file_watcher_checkpoint(_empty_checkpoint(tmp_path)).decode(
        "utf-8"
    )
    for forbidden in (
        "first_pending_at",
        "last_change_at",
        "last_observed_monotonic",
        "task_id",
        "provider",
        "run_id",
        "correlation_id",
    ):
        assert forbidden not in encoded


def test_encoding_contains_no_file_content(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    current = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=current)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(current,),
        pending_changes=(change,),
    )
    encoded = encode_file_watcher_checkpoint(checkpoint).decode("utf-8")
    for forbidden in ("content", "chunks", "embedding", "payload_base64"):
        assert forbidden not in encoded


# --- Decode failures ---


def test_decode_invalid_utf8() -> None:
    with pytest.raises(RuntimeError, match="^checkpoint_invalid_encoding$"):
        decode_file_watcher_checkpoint(b"\xff\xfe")


def test_decode_malformed_json() -> None:
    with pytest.raises(RuntimeError, match="^checkpoint_invalid_json$"):
        decode_file_watcher_checkpoint(b"{not-json\n")


def test_decode_top_level_array() -> None:
    with pytest.raises(RuntimeError, match="^checkpoint_invalid$"):
        decode_file_watcher_checkpoint(b"[]\n")


def test_decode_unknown_schema_version(tmp_path: Path) -> None:
    payload = {
        "schema_version": "lkw.file_watcher_checkpoint.v999",
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": [str(tmp_path.resolve())],
    }
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    with pytest.raises(RuntimeError, match="^checkpoint_invalid$") as exc_info:
        decode_file_watcher_checkpoint(raw)
    assert "schema" not in str(exc_info.value).lower() or str(exc_info.value) == (
        "checkpoint_invalid"
    )


def test_decode_missing_identity(tmp_path: Path) -> None:
    payload = {
        "schema_version": "lkw.file_watcher_checkpoint.v1",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": [str(tmp_path.resolve())],
    }
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    with pytest.raises(RuntimeError, match="^checkpoint_invalid$"):
        decode_file_watcher_checkpoint(raw)


def test_decode_invalid_nested_snapshot(tmp_path: Path) -> None:
    payload = {
        "schema_version": "lkw.file_watcher_checkpoint.v1",
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": [str(tmp_path.resolve())],
        "baseline_snapshots": [
            {
                "schema_version": "lkw.file_snapshot.v1",
                "path": str(_abs(tmp_path, "a.txt")),
                "size_bytes": -1,
                "modified_time_ns": 1,
            }
        ],
    }
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    with pytest.raises(RuntimeError, match="^checkpoint_invalid$"):
        decode_file_watcher_checkpoint(raw)


def test_decode_invalid_nested_change(tmp_path: Path) -> None:
    path = str(_abs(tmp_path, "a.txt"))
    payload = {
        "schema_version": "lkw.file_watcher_checkpoint.v1",
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": [str(tmp_path.resolve())],
        "pending_changes": [
            {
                "schema_version": "lkw.file_change.v1",
                "kind": "created",
                "path": path,
                "current": None,
            }
        ],
    }
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    with pytest.raises(RuntimeError, match="^checkpoint_invalid$"):
        decode_file_watcher_checkpoint(raw)


# --- Store ---


def test_store_load_missing_file(tmp_path: Path) -> None:
    store = JsonFileWatcherCheckpointStore(tmp_path / "missing" / "checkpoint.json")
    assert store.load() is None


def test_store_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "data" / "file_watcher" / "checkpoint.json"
    store = JsonFileWatcherCheckpointStore(target)
    checkpoint = _empty_checkpoint(tmp_path)
    store.save(checkpoint)
    assert target.is_file()
    assert store.load() == checkpoint


def test_store_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.json"
    store = JsonFileWatcherCheckpointStore(target)
    first = _empty_checkpoint(tmp_path, tenant_id="tenant-a")
    second = _empty_checkpoint(tmp_path, tenant_id="tenant-b")
    store.save(first)
    store.save(second)
    assert store.load() == second
    leftovers = list(target.parent.glob("lkw-file-watcher-checkpoint.*.tmp"))
    assert leftovers == []


def test_atomic_replace_failure_preserves_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "checkpoint.json"
    store = JsonFileWatcherCheckpointStore(target)
    first = _empty_checkpoint(tmp_path, tenant_id="tenant-a")
    second = _empty_checkpoint(tmp_path, tenant_id="tenant-b")
    store.save(first)
    previous_bytes = target.read_bytes()

    def _boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", _boom)
    with pytest.raises(RuntimeError, match="^checkpoint_write_failed$"):
        store.save(second)
    assert target.read_bytes() == previous_bytes
    leftovers = list(target.parent.glob("lkw-file-watcher-checkpoint.*.tmp"))
    assert leftovers == []


def test_relative_checkpoint_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="^checkpoint_path_must_be_absolute$"):
        JsonFileWatcherCheckpointStore("relative/checkpoint.json")


def test_checkpoint_path_directory_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "as-dir"
    directory.mkdir()
    store = JsonFileWatcherCheckpointStore(directory)
    with pytest.raises(RuntimeError, match="^checkpoint_not_file$"):
        store.load()


def test_checkpoint_read_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "checkpoint.json"
    target.write_bytes(b"{}\n")
    store = JsonFileWatcherCheckpointStore(target)

    def _boom(self: Path) -> bytes:
        raise OSError("read boom")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    with pytest.raises(RuntimeError, match="^checkpoint_read_failed$") as exc_info:
        store.load()
    assert "read boom" not in str(exc_info.value)


def test_checkpoint_write_failed_on_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "checkpoint.json"
    store = JsonFileWatcherCheckpointStore(target)

    def _boom(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("mkdir boom")

    monkeypatch.setattr(Path, "mkdir", _boom)
    with pytest.raises(RuntimeError, match="^checkpoint_write_failed$") as exc_info:
        store.save(_empty_checkpoint(tmp_path))
    assert "mkdir boom" not in str(exc_info.value)


# --- data_home path ---


def test_file_watcher_checkpoint_path_absolute(tmp_path: Path) -> None:
    home = tmp_path.resolve()
    resolved = file_watcher_checkpoint_path(home)
    assert resolved == (home / "data" / "file_watcher" / "checkpoint.json").resolve()


def test_blank_data_home_rejected() -> None:
    with pytest.raises(RuntimeError, match="^data_home_must_be_non_blank$"):
        file_watcher_checkpoint_path("   ")


def test_relative_data_home_rejected() -> None:
    with pytest.raises(RuntimeError, match="^data_home_must_be_absolute$"):
        file_watcher_checkpoint_path("relative/home")


def test_data_home_user_marker_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    resolved = file_watcher_checkpoint_path("~/data-home")
    assert resolved.is_absolute()
    assert resolved.name == "checkpoint.json"
    assert resolved.parent.name == "file_watcher"


# --- Runtime export ---


def test_export_before_initialization_rejected(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path, snapshots=[()])
    with pytest.raises(RuntimeError, match="^file_watcher_not_initialized$"):
        runtime.export_checkpoint()


def test_export_initialized_baseline(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    runtime, _, _ = _runtime(tmp_path, snapshots=[(snap,)])
    runtime.initialize()
    checkpoint = runtime.export_checkpoint()
    assert checkpoint.baseline_snapshots == (snap,)
    assert checkpoint.pending_changes == ()


def test_export_pending_changes_sorted(tmp_path: Path) -> None:
    path_a = _abs(tmp_path, "a.txt")
    path_z = _abs(tmp_path, "z.txt")
    snap_a = _snap(path_a, size=1, mtime=1)
    snap_z = _snap(path_z, size=1, mtime=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (snap_z, snap_a)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    checkpoint = runtime.export_checkpoint()
    keys = [normalize_watch_path_key(c.path) for c in checkpoint.pending_changes]
    assert keys == sorted(keys)
    assert checkpoint.tenant_id == "tenant-a"
    assert checkpoint.workspace_id == "workspace-a"
    assert checkpoint.collection_id == "collection-a"


def test_export_independent_of_later_mutation(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    v2 = _snap(path, size=2, mtime=2)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (v1,), (v2,)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    exported = runtime.export_checkpoint()
    runtime.poll_once(now_monotonic=2.0)
    assert exported.baseline_snapshots == (v1,)
    assert exported.pending_changes[0].current == v1


def test_timing_changes_do_not_alter_encoded_checkpoint(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (snap,)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    first = encode_file_watcher_checkpoint(runtime.export_checkpoint())
    runtime.flush_if_due(now_monotonic=2.0)
    second = encode_file_watcher_checkpoint(runtime.export_checkpoint())
    assert first == second


# --- Runtime restore ---


def test_restore_into_uninitialized_runtime(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(snap,),
        pending_changes=(),
    )
    runtime, _, _ = _runtime(tmp_path, snapshots=[()])
    assert runtime.initialized is False
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    assert runtime.initialized is True
    assert runtime.baseline_file_count == 1
    assert runtime.pending_change_count == 0


def test_restore_replaces_initialized_baseline_and_pending(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    old = _snap(path, size=1, mtime=1)
    new = _snap(path, size=9, mtime=9)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(old,), (new,)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=5.0)
    assert runtime.pending_change_count == 1

    restored_snap = _snap(path, size=3, mtime=3)
    change = FileChange(kind="created", path=str(path), current=restored_snap)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(restored_snap,),
        pending_changes=(change,),
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    exported = runtime.export_checkpoint()
    assert exported.baseline_snapshots == (restored_snap,)
    assert exported.pending_changes[0].current == restored_snap


def test_restore_empty_pending_resets_timing(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (snap,)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=5.0)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(snap,),
        pending_changes=(),
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=2.0)
    result = runtime.flush_if_due(now_monotonic=2.0)
    assert result.status == "idle"


def test_restore_pending_starts_new_debounce(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=snap)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(snap,),
        pending_changes=(change,),
    )
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(snap,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=100.0)
    pending = runtime.flush_if_due(now_monotonic=100.0)
    assert pending.status == "pending"
    due = runtime.flush_if_due(now_monotonic=101.0)
    assert due.status == "enqueued"
    assert len(queue.jobs) == 1


def test_restore_accepts_lower_new_process_time(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (snap,)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=5000.0)
    checkpoint = runtime.export_checkpoint()
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    assert runtime.initialized is True


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True, "1"])
def test_restore_rejects_invalid_monotonic(tmp_path: Path, bad: object) -> None:
    checkpoint = _empty_checkpoint(tmp_path)
    runtime, _, _ = _runtime(tmp_path, snapshots=[()])
    with pytest.raises(RuntimeError, match="^invalid_monotonic_time$"):
        runtime.restore_checkpoint(checkpoint, now_monotonic=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "other-tenant"),
        ("workspace_id", "other-workspace"),
        ("collection_id", "other-collection"),
    ],
)
def test_identity_mismatch_fields(tmp_path: Path, field: str, value: str) -> None:
    kwargs = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": frozenset({str(tmp_path.resolve())}),
        "baseline_snapshots": (),
        "pending_changes": (),
    }
    kwargs[field] = value
    checkpoint = build_file_watcher_checkpoint(**kwargs)  # type: ignore[arg-type]
    runtime, _, _ = _runtime(tmp_path, snapshots=[()])
    runtime.initialize()
    with pytest.raises(RuntimeError, match="^checkpoint_identity_mismatch$") as exc:
        runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    assert value not in str(exc.value)


def test_identity_mismatch_allowed_roots(tmp_path: Path) -> None:
    other = (tmp_path / "other").resolve()
    other.mkdir()
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(other)}),
        baseline_snapshots=(),
        pending_changes=(),
    )
    runtime, _, _ = _runtime(tmp_path, snapshots=[()])
    runtime.initialize()
    with pytest.raises(RuntimeError, match="^checkpoint_identity_mismatch$") as exc:
        runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    assert str(other) not in str(exc.value)


def test_policy_fields_do_not_block_restore(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=snap)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(snap,),
        pending_changes=(change,),
    )
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[()],
        debounce_seconds=3.0,
        max_batch_wait_seconds=9.0,
        priority="high",
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    assert runtime.pending_change_count == 1


def test_failed_restore_preserves_previous_state(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (snap,)],
        debounce_seconds=10.0,
        max_batch_wait_seconds=30.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=5.0)
    before = runtime.export_checkpoint()
    before_pending = runtime.pending_change_count

    bad = build_file_watcher_checkpoint(
        tenant_id="other",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(),
        pending_changes=(),
    )
    with pytest.raises(RuntimeError, match="^checkpoint_identity_mismatch$"):
        runtime.restore_checkpoint(bad, now_monotonic=1.0)
    assert runtime.export_checkpoint() == before
    assert runtime.pending_change_count == before_pending


# --- Downtime recovery ---


def test_downtime_created(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    checkpoint = _empty_checkpoint(tmp_path)
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(snap,), (snap,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    first = runtime.poll_once(now_monotonic=1.0)
    assert first.detected_change_count == 1
    assert first.status == "pending"
    assert first.pending_change_count == 1
    second = runtime.poll_once(now_monotonic=2.0)
    assert second.status == "enqueued"
    assert len(queue.jobs) == 1
    assert queue.jobs[0].source_paths == (str(path),)  # type: ignore[attr-defined]


def test_downtime_modified(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    v2 = _snap(path, size=2, mtime=2)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(v1,),
        pending_changes=(),
    )
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(v2,), (v2,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    first = runtime.poll_once(now_monotonic=1.0)
    assert first.detected_change_count == 1
    assert first.status == "pending"
    second = runtime.poll_once(now_monotonic=2.0)
    assert second.status == "enqueued"
    assert second.change_token == file_change_token((v2,))
    assert queue.jobs[0].change_token == file_change_token((v2,))  # type: ignore[attr-defined]


def test_downtime_deleted(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(v1,),
        pending_changes=(),
    )
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(), ()],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    first = runtime.poll_once(now_monotonic=1.0)
    assert first.detected_change_count == 1
    assert first.status == "pending"
    second = runtime.poll_once(now_monotonic=2.0)
    assert second.status == "deletions_only"
    assert queue.jobs == []


def test_restored_pending_updated_during_downtime(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    v2 = _snap(path, size=2, mtime=2)
    change = FileChange(kind="created", path=str(path), current=v1)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(v1,),
        pending_changes=(change,),
    )
    old_token = file_change_token((v1,))
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(v2,), (v2,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    first = runtime.poll_once(now_monotonic=1.0)
    assert first.pending_change_count == 1
    second = runtime.poll_once(now_monotonic=2.0)
    assert second.status == "enqueued"
    assert second.change_token == file_change_token((v2,))
    assert second.change_token != old_token
    assert queue.jobs[0].change_token == file_change_token((v2,))  # type: ignore[attr-defined]


def test_restored_pending_deleted_during_downtime(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=v1)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(v1,),
        pending_changes=(change,),
    )
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(), ()],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    first = runtime.poll_once(now_monotonic=1.0)
    assert first.pending_change_count == 1
    second = runtime.poll_once(now_monotonic=2.0)
    assert second.status == "deletions_only"
    assert queue.jobs == []


def test_restored_deletion_recreated_during_downtime(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    previous = _snap(path, size=1, mtime=1)
    created = _snap(path, size=2, mtime=2)
    change = FileChange(kind="deleted", path=str(path), previous=previous)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(),
        pending_changes=(change,),
    )
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(created,), (created,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.restore_checkpoint(checkpoint, now_monotonic=1.0)
    first = runtime.poll_once(now_monotonic=1.0)
    assert first.status == "pending"
    second = runtime.poll_once(now_monotonic=2.0)
    assert second.status == "enqueued"
    assert len(queue.jobs) == 1
    assert queue.jobs[0].source_paths == (str(created.path),)  # type: ignore[attr-defined]


def test_retry_identity_across_restart(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(), (snap,), (snap,)],
        enqueuer=_FakeEnqueuer(fail_times=1),
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    failed = runtime.poll_once(now_monotonic=2.0)
    assert failed.status == "enqueue_failed"
    pre_job = queue.jobs[0]
    pre_token = pre_job.change_token  # type: ignore[attr-defined]
    pre_paths = pre_job.source_paths  # type: ignore[attr-defined]
    pre_key = background_ingest_idempotency_key(pre_job)  # type: ignore[arg-type]

    store = JsonFileWatcherCheckpointStore(tmp_path / "checkpoint.json")
    store.save(runtime.export_checkpoint())

    restored, _, queue2 = _runtime(
        tmp_path,
        snapshots=[(snap,), (snap,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    assert restore_file_watcher_runtime(
        runtime=restored, store=store, now_monotonic=1.0
    )
    first = restored.poll_once(now_monotonic=1.0)
    assert first.status == "pending"
    second = restored.poll_once(now_monotonic=2.0)
    assert second.status == "enqueued"
    job = queue2.jobs[0]
    assert job.change_token == pre_token  # type: ignore[attr-defined]
    assert job.source_paths == pre_paths  # type: ignore[attr-defined]
    assert background_ingest_idempotency_key(job) == pre_key  # type: ignore[arg-type]


def test_newer_version_identity_across_restart(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    v2 = _snap(path, size=2, mtime=2)
    runtime, _, queue = _runtime(
        tmp_path,
        snapshots=[(), (v1,), (v1,)],
        enqueuer=_FakeEnqueuer(fail_times=99),
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    failed = runtime.poll_once(now_monotonic=2.0)
    assert failed.status == "enqueue_failed"
    pre_job = queue.jobs[0]
    pre_token = pre_job.change_token  # type: ignore[attr-defined]
    pre_key = background_ingest_idempotency_key(pre_job)  # type: ignore[arg-type]

    store = JsonFileWatcherCheckpointStore(tmp_path / "checkpoint.json")
    store.save(runtime.export_checkpoint())

    restored, _, queue2 = _runtime(
        tmp_path,
        snapshots=[(v2,), (v2,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    assert restore_file_watcher_runtime(
        runtime=restored, store=store, now_monotonic=1.0
    )
    restored.poll_once(now_monotonic=1.0)
    result = restored.poll_once(now_monotonic=2.0)
    assert result.status == "enqueued"
    job = queue2.jobs[0]
    assert job.change_token != pre_token  # type: ignore[attr-defined]
    assert background_ingest_idempotency_key(job) != pre_key  # type: ignore[arg-type]
    assert job.change_token == file_change_token((v2,))  # type: ignore[attr-defined]


def test_missing_checkpoint_helper_leaves_runtime_unchanged(tmp_path: Path) -> None:
    store = JsonFileWatcherCheckpointStore(tmp_path / "missing.json")
    runtime, _, _ = _runtime(tmp_path, snapshots=[()])
    assert (
        restore_file_watcher_runtime(runtime=runtime, store=store, now_monotonic=1.0)
        is False
    )
    assert runtime.initialized is False


def test_invalid_checkpoint_helper_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.json"
    target.write_bytes(b"{not-json\n")
    store = JsonFileWatcherCheckpointStore(target)
    runtime, provider, _ = _runtime(tmp_path, snapshots=[()])
    with pytest.raises(RuntimeError, match="^checkpoint_invalid_json$"):
        restore_file_watcher_runtime(runtime=runtime, store=store, now_monotonic=1.0)
    assert runtime.initialized is False
    assert provider.calls == []


def test_safe_checkpoint_content_fields(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    path = (root / "a.txt").resolve()
    snap = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=snap)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(root)}),
        baseline_snapshots=(snap,),
        pending_changes=(change,),
    )
    encoded = encode_file_watcher_checkpoint(checkpoint).decode("utf-8")
    payload = json.loads(encoded)

    def _collect_keys(value: object, keys: set[str]) -> None:
        if isinstance(value, dict):
            keys.update(str(key) for key in value)
            for nested in value.values():
                _collect_keys(nested, keys)
        elif isinstance(value, list):
            for nested in value:
                _collect_keys(nested, keys)

    keys: set[str] = set()
    _collect_keys(payload, keys)
    for forbidden in (
        "first_pending_at",
        "last_change_at",
        "last_observed_monotonic",
        "task_id",
        "provider",
        "run_id",
        "correlation_id",
        "payload_base64",
        "content",
        "chunks",
        "embedding",
        "credentials",
        "broker_url",
    ):
        assert forbidden not in keys
        assert f'"{forbidden}"' not in encoded


def test_boundary_static_inspection() -> None:
    checkpoint_src = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    runtime_src = Path(runtime_module.__file__).read_text(encoding="utf-8")
    combined = checkpoint_src + "\n" + runtime_src
    for required in (
        "os.replace",
        "os.fsync",
        "mkstemp",
        "build_file_watcher_checkpoint",
        "restore_checkpoint",
    ):
        assert required in combined
    for forbidden in (
        "time.sleep",
        "time.monotonic",
        "while True",
        "threading",
        "asyncio",
        "signal.signal",
        "Kafka",
        "Redis",
        "message_bus_enqueue",
        "TaskRequest(",
        "TaskHandle(",
        "ProofReceipt",
        "MongoClient",
        "pymongo",
        "fcntl",
        "msvcrt",
        "portalocker",
    ):
        assert forbidden not in combined


def test_build_ingest_job_from_restored_pending_identity(tmp_path: Path) -> None:
    """Helper sanity: derived identity comes from pending snapshots only."""
    path = _abs(tmp_path, "a.txt")
    snap = _snap(path, size=1, mtime=1)
    change = FileChange(kind="created", path=str(path), current=snap)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(snap,),
        pending_changes=(change,),
    )
    batch = build_incremental_file_change_batch(checkpoint.pending_changes)
    job = build_file_watcher_ingest_job(
        batch,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
    )
    assert job.change_token == batch.change_token
    assert background_ingest_idempotency_key(job)
