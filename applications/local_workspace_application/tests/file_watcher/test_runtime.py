# © Artur Czarnecki. All rights reserved.

"""Tests for LKW.7B1 file-watcher runtime state machine."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueOutput
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
    decode_background_ingest_job,
)
from local_workspace_application.file_watcher import (
    BackgroundIngestEnqueuer,
    FileSnapshot,
    FileSnapshotProvider,
    FileWatcherCycleResult,
    FileWatcherRuntime,
    FileWatcherRuntimeConfig,
    build_file_watcher_runtime,
    build_incremental_file_change_batch,
    file_change_token,
)
from local_workspace_application.file_watcher.contracts import FileChange

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
    def __init__(
        self,
        *,
        fail_times: int = 0,
        invalid_output: bool = False,
        output: MessageBusEnqueueOutput | None = None,
    ) -> None:
        self.jobs: list[LkwBackgroundIngestJob] = []
        self._fail_times = fail_times
        self._invalid_output = invalid_output
        self._output = output
        self._calls = 0

    def __call__(self, job: LkwBackgroundIngestJob) -> MessageBusEnqueueOutput:
        self.jobs.append(job)
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("enqueue boom")
        if self._invalid_output:
            return MessageBusEnqueueOutput(task_id="", provider="")
        if self._output is not None:
            return self._output
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


class _FakeMessageBus(TaskQueue):
    def __init__(self) -> None:
        self.requests: list[TaskRequest] = []

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        self.requests.append(request)
        return TaskHandle(
            task_id=request.run_id.strip(),
            provider="fake",
            tenant_id=request.tenant_id,
        )

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        return TaskStatus.PENDING

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        return None


def test_initial_snapshot_becomes_baseline_without_created_or_enqueue(
    tmp_path: Path,
) -> None:
    path = _abs(tmp_path, "existing.txt")
    baseline = (_snap(path, size=1, mtime=10),)
    runtime, provider, enqueuer = _runtime(tmp_path, snapshots=[baseline, baseline])

    returned = runtime.initialize()

    assert returned == baseline
    assert runtime.initialized is True
    assert runtime.baseline_file_count == 1
    assert runtime.pending_change_count == 0
    assert enqueuer.jobs == []

    result = runtime.poll_once(now_monotonic=1.0)
    assert result.status == "idle"
    assert result.detected_change_count == 0
    assert enqueuer.jobs == []
    assert len(provider.calls) == 2


def test_initialize_performs_no_enqueue(tmp_path: Path) -> None:
    runtime, _, enqueuer = _runtime(tmp_path, snapshots=[()])
    runtime.initialize()
    assert enqueuer.jobs == []


def test_reinitialize_replaces_baseline_and_clears_pending(tmp_path: Path) -> None:
    a = _abs(tmp_path, "a.txt")
    b = _abs(tmp_path, "b.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(a, size=1, mtime=1),),
            (_snap(b, size=2, mtime=2),),
        ],
    )
    runtime.initialize()
    pending = runtime.poll_once(now_monotonic=1.0)
    assert pending.status == "pending"
    assert runtime.pending_change_count == 1

    reset = runtime.initialize()
    assert reset == (_snap(b, size=2, mtime=2),)
    assert runtime.baseline_file_count == 1
    assert runtime.pending_change_count == 0
    assert enqueuer.jobs == []


def test_failed_first_initialization_leaves_uninitialized(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[RuntimeError("watch_root_not_found")],
    )
    with pytest.raises(RuntimeError, match="watch_root_not_found"):
        runtime.initialize()
    assert runtime.initialized is False
    assert runtime.baseline_file_count == 0


def test_failed_reinitialization_preserves_previous_state(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(path, size=1, mtime=1),),
            RuntimeError("file_snapshot_failed"),
        ],
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    assert runtime.pending_change_count == 1
    assert runtime.baseline_file_count == 1

    with pytest.raises(RuntimeError, match="file_snapshot_failed"):
        runtime.initialize()

    assert runtime.initialized is True
    assert runtime.baseline_file_count == 1
    assert runtime.pending_change_count == 1
    assert enqueuer.jobs == []


def test_unchanged_poll_returns_idle(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    snap = (_snap(path, size=1, mtime=1),)
    runtime, _, _ = _runtime(tmp_path, snapshots=[snap, snap])
    runtime.initialize()
    result = runtime.poll_once(now_monotonic=1.0)
    assert result.status == "idle"
    assert result.pending_change_count == 0


def test_new_file_before_debounce_returns_pending(tmp_path: Path) -> None:
    path = _abs(tmp_path, "new.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[(), (_snap(path, size=1, mtime=1),)],
    )
    runtime.initialize()
    result = runtime.poll_once(now_monotonic=1.0)
    assert result.status == "pending"
    assert result.detected_change_count == 1
    assert result.pending_change_count == 1
    assert enqueuer.jobs == []


def test_modified_file_before_debounce_returns_pending(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[
            (_snap(path, size=1, mtime=1),),
            (_snap(path, size=2, mtime=2),),
        ],
    )
    runtime.initialize()
    result = runtime.poll_once(now_monotonic=1.0)
    assert result.status == "pending"
    assert result.detected_change_count == 1


def test_pending_count_is_path_count_not_event_count(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(path, size=1, mtime=1),),
            (_snap(path, size=2, mtime=2),),
            (_snap(path, size=3, mtime=3),),
        ],
        debounce_seconds=5.0,
        max_batch_wait_seconds=20.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    runtime.poll_once(now_monotonic=2.0)
    result = runtime.poll_once(now_monotonic=3.0)
    assert result.status == "pending"
    assert result.pending_change_count == 1
    assert runtime.pending_change_count == 1


def test_duplicate_final_change_for_one_path_remains_one_pending(
    tmp_path: Path,
) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=9, mtime=9)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (final,), (final,)],
        debounce_seconds=5.0,
        max_batch_wait_seconds=20.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    result = runtime.poll_once(now_monotonic=1.5)
    assert result.status == "pending"
    assert result.pending_change_count == 1
    assert result.detected_change_count == 0


def test_quiet_debounce_enqueues_once(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=1, mtime=1)
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[(), (final,)],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.initialize()
    assert runtime.poll_once(now_monotonic=1.0).status == "pending"
    assert runtime.flush_if_due(now_monotonic=1.0 + 1.0 - 0.001).status == "pending"
    result = runtime.flush_if_due(now_monotonic=1.0 + 1.0)
    assert result.status == "enqueued"
    assert len(enqueuer.jobs) == 1
    assert enqueuer.jobs[0].change_token == file_change_token((final,))


def test_maximum_wait_enqueues_during_continuous_writes(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(path, size=1, mtime=1),),
            (_snap(path, size=2, mtime=2),),
            (_snap(path, size=3, mtime=3),),
            (_snap(path, size=4, mtime=4),),
        ],
        debounce_seconds=2.0,
        max_batch_wait_seconds=5.0,
    )
    runtime.initialize()
    assert runtime.poll_once(now_monotonic=1.0).status == "pending"
    assert runtime.poll_once(now_monotonic=2.5).status == "pending"
    assert runtime.poll_once(now_monotonic=4.0).status == "pending"
    result = runtime.poll_once(now_monotonic=6.0)
    assert result.status == "enqueued"
    assert len(enqueuer.jobs) == 1
    assert enqueuer.jobs[0].source_paths == (str(path),)
    assert enqueuer.jobs[0].change_token == file_change_token(
        (_snap(path, size=4, mtime=4),)
    )


def test_multi_path_batch_single_enqueue(tmp_path: Path) -> None:
    a = _abs(tmp_path, "a.txt")
    b = _abs(tmp_path, "b.txt")
    snaps = (_snap(a, size=1, mtime=1), _snap(b, size=2, mtime=2))
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[(), snaps],
        debounce_seconds=1.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    result = runtime.flush_if_due(now_monotonic=2.0)
    assert result.status == "enqueued"
    assert result.actionable_path_count == 2
    assert len(enqueuer.jobs) == 1
    job = enqueuer.jobs[0]
    assert job.source_paths == tuple(sorted((str(a), str(b))))
    assert job.change_token == file_change_token(snaps)


def test_created_then_modified_enqueues_final_modified(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=9, mtime=9)
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(path, size=1, mtime=1),),
            (final,),
        ],
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    runtime.poll_once(now_monotonic=1.5)
    result = runtime.flush_if_due(now_monotonic=2.5)
    assert result.status == "enqueued"
    assert enqueuer.jobs[0].change_token == file_change_token((final,))


def test_modified_then_modified_enqueues_last_snapshot(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    last = _snap(path, size=3, mtime=30)
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (_snap(path, size=1, mtime=1),),
            (_snap(path, size=2, mtime=2),),
            (last,),
        ],
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    runtime.poll_once(now_monotonic=1.5)
    result = runtime.flush_if_due(now_monotonic=2.5)
    assert result.status == "enqueued"
    assert enqueuer.jobs[0].change_token == file_change_token((last,))


def test_created_then_deleted_is_deletions_only(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(path, size=1, mtime=1),),
            (),
        ],
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    runtime.poll_once(now_monotonic=1.5)
    result = runtime.flush_if_due(now_monotonic=2.5)
    assert result.status == "deletions_only"
    assert result.deleted_path_count == 1
    assert result.change_token is None
    assert enqueuer.jobs == []
    assert runtime.pending_change_count == 0


def test_modified_then_deleted_is_deletions_only(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (_snap(path, size=1, mtime=1),),
            (_snap(path, size=2, mtime=2),),
            (),
        ],
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    runtime.poll_once(now_monotonic=1.5)
    result = runtime.flush_if_due(now_monotonic=2.5)
    assert result.status == "deletions_only"
    assert enqueuer.jobs == []


def test_deleted_then_created_enqueues_actionable_final(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=5, mtime=50)
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (_snap(path, size=1, mtime=1),),
            (),
            (final,),
        ],
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    runtime.poll_once(now_monotonic=1.5)
    result = runtime.flush_if_due(now_monotonic=2.5)
    assert result.status == "enqueued"
    assert enqueuer.jobs[0].change_token == file_change_token((final,))


def test_mixed_actionable_and_deletion(tmp_path: Path) -> None:
    keep = _abs(tmp_path, "keep.txt")
    gone = _abs(tmp_path, "gone.txt")
    runtime, _, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (
                _snap(keep, size=1, mtime=1),
                _snap(gone, size=1, mtime=1),
            ),
            (_snap(keep, size=2, mtime=2),),
        ],
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    result = runtime.flush_if_due(now_monotonic=2.0)
    assert result.status == "enqueued"
    assert result.actionable_path_count == 1
    assert result.deleted_path_count == 1
    assert enqueuer.jobs[0].source_paths == (str(keep),)
    assert str(gone) not in enqueuer.jobs[0].source_paths
    assert runtime.pending_change_count == 0


def test_enqueue_failure_retains_pending_and_retries_same_identity(
    tmp_path: Path,
) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=1, mtime=1)
    enqueuer = _FakeEnqueuer(fail_times=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (final,)],
        enqueuer=enqueuer,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    failed = runtime.flush_if_due(now_monotonic=2.0)
    assert failed.status == "enqueue_failed"
    assert failed.error_id == "background_ingest_enqueue_failed"
    assert failed.change_token == file_change_token((final,))
    assert runtime.pending_change_count == 1
    failed_job = enqueuer.jobs[0]
    failed_key = background_ingest_idempotency_key(failed_job)

    success = runtime.flush_if_due(now_monotonic=2.0)
    assert success.status == "enqueued"
    assert len(enqueuer.jobs) == 2
    assert enqueuer.jobs[1].change_token == failed_job.change_token
    assert enqueuer.jobs[1].source_paths == failed_job.source_paths
    assert background_ingest_idempotency_key(enqueuer.jobs[1]) == failed_key
    assert runtime.pending_change_count == 0


def test_failure_followed_by_newer_version_enqueues_version_two(
    tmp_path: Path,
) -> None:
    path = _abs(tmp_path, "a.txt")
    v1 = _snap(path, size=1, mtime=1)
    v2 = _snap(path, size=2, mtime=2)
    enqueuer = _FakeEnqueuer(fail_times=1)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (v1,), (v2,)],
        enqueuer=enqueuer,
        debounce_seconds=1.0,
        max_batch_wait_seconds=10.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    failed = runtime.flush_if_due(now_monotonic=2.0)
    assert failed.status == "enqueue_failed"
    token_v1 = failed.change_token
    key_v1 = background_ingest_idempotency_key(enqueuer.jobs[0])

    pending = runtime.poll_once(now_monotonic=2.5)
    assert pending.status == "pending"
    assert runtime.pending_change_count == 1

    success = runtime.flush_if_due(now_monotonic=3.5)
    assert success.status == "enqueued"
    assert success.change_token == file_change_token((v2,))
    assert success.change_token != token_v1
    assert background_ingest_idempotency_key(enqueuer.jobs[1]) != key_v1
    assert enqueuer.jobs[1].change_token == file_change_token((v2,))


def test_invalid_enqueue_output_is_enqueue_failed(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    enqueuer = _FakeEnqueuer(invalid_output=True)
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (_snap(path, size=1, mtime=1),)],
        enqueuer=enqueuer,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    result = runtime.flush_if_due(now_monotonic=2.0)
    assert result.status == "enqueue_failed"
    assert result.error_id == "background_ingest_enqueue_failed"
    assert runtime.pending_change_count == 1


def test_snapshot_failure_preserves_state_and_time(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    runtime, provider, enqueuer = _runtime(
        tmp_path,
        snapshots=[
            (),
            (_snap(path, size=1, mtime=1),),
            RuntimeError("file_snapshot_failed"),
            (_snap(path, size=1, mtime=1),),
        ],
        debounce_seconds=5.0,
        max_batch_wait_seconds=20.0,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    assert runtime.pending_change_count == 1
    baseline_before = runtime.baseline_file_count

    with pytest.raises(RuntimeError, match="file_snapshot_failed"):
        runtime.poll_once(now_monotonic=1.5)

    assert runtime.baseline_file_count == baseline_before
    assert runtime.pending_change_count == 1
    assert enqueuer.jobs == []

    # Failed poll did not accept time, so the same timestamp may be retried.
    result = runtime.poll_once(now_monotonic=1.5)
    assert result.status == "pending"
    assert len(provider.calls) == 4


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1.0, "invalid_monotonic_time"),
        (math.nan, "invalid_monotonic_time"),
        (math.inf, "invalid_monotonic_time"),
        (-math.inf, "invalid_monotonic_time"),
    ],
)
def test_invalid_monotonic_time_rejected(
    tmp_path: Path,
    value: float,
    error: str,
) -> None:
    runtime, _, enqueuer = _runtime(tmp_path, snapshots=[(), ()])
    runtime.initialize()
    with pytest.raises(RuntimeError, match=error):
        runtime.poll_once(now_monotonic=value)
    assert runtime.pending_change_count == 0
    assert enqueuer.jobs == []


def test_backward_timestamp_rejected(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path, snapshots=[(), (), ()])
    runtime.initialize()
    runtime.poll_once(now_monotonic=5.0)
    with pytest.raises(RuntimeError, match="monotonic_time_regressed"):
        runtime.poll_once(now_monotonic=4.0)
    assert runtime.pending_change_count == 0


def test_requires_initialization_before_poll_or_flush(tmp_path: Path) -> None:
    provider = _FakeSnapshotProvider([()])
    enqueuer = _FakeEnqueuer()
    runtime = FileWatcherRuntime(
        config=_config(tmp_path),
        snapshot_provider=provider,
        enqueuer=enqueuer,
    )
    with pytest.raises(RuntimeError, match="file_watcher_not_initialized"):
        runtime.poll_once(now_monotonic=1.0)
    with pytest.raises(RuntimeError, match="file_watcher_not_initialized"):
        runtime.flush_if_due(now_monotonic=1.0)
    assert provider.calls == []
    assert enqueuer.jobs == []


def test_cycle_result_is_safe_for_logs(tmp_path: Path) -> None:
    path = _abs(tmp_path, "secret-file.txt")
    runtime, _, _ = _runtime(
        tmp_path,
        snapshots=[(), (_snap(path, size=1, mtime=1),)],
    )
    runtime.initialize()
    pending = runtime.poll_once(now_monotonic=1.0)
    enqueued = runtime.flush_if_due(now_monotonic=2.0)
    for result in (pending, enqueued):
        payload = result.model_dump_json()
        assert str(path) not in payload
        assert "secret-file" not in payload
        assert "payload_base64" not in payload
        assert "traceback" not in payload.lower()
        assert "broker_url" not in payload.lower()
        assert "credential" not in payload.lower()
        assert "Exception" not in payload


def test_inconsistent_cycle_result_rejected() -> None:
    with pytest.raises(ValidationError):
        FileWatcherCycleResult(status="idle", pending_change_count=1)
    with pytest.raises(ValidationError):
        FileWatcherCycleResult(
            status="enqueued",
            pending_change_count=0,
            actionable_path_count=1,
            change_token="sha256:" + ("a" * 64),
            task_id="t",
            provider="p",
            error_id="background_ingest_enqueue_failed",
        )


def test_build_file_watcher_runtime_uses_existing_enqueue_helper(
    tmp_path: Path,
) -> None:
    path = _abs(tmp_path, "a.txt")
    final = _snap(path, size=1, mtime=1)
    provider = _FakeSnapshotProvider([(), (final,)])
    bus = _FakeMessageBus()
    runtime = build_file_watcher_runtime(
        config=_config(tmp_path),
        wiring_context=ToolWiringContext(message_bus=bus),
        snapshot_provider=provider,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    result = runtime.flush_if_due(now_monotonic=2.0)
    assert result.status == "enqueued"
    assert len(bus.requests) == 1
    request = bus.requests[0]
    job = decode_background_ingest_job(request.payload)
    key = background_ingest_idempotency_key(job)
    assert result.broker_run_id == key
    assert result.idempotency_key == key
    assert request.task_name == LKW_BACKGROUND_INGEST_TASK_NAME
    assert request.tenant_id == "tenant-a"
    assert job.requested_by == "lkw.file_watcher"
    assert job.reason == "lkw.7.incremental_change"
    assert job.change_token == file_change_token((final,))
    assert request.idempotency_key == key
    assert request.run_id == key


def test_missing_message_bus_returns_safe_enqueue_failed(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    provider = _FakeSnapshotProvider([(), (_snap(path, size=1, mtime=1),)])
    runtime = build_file_watcher_runtime(
        config=_config(tmp_path),
        wiring_context=ToolWiringContext(message_bus=None),
        snapshot_provider=provider,
    )
    runtime.initialize()
    runtime.poll_once(now_monotonic=1.0)
    result = runtime.flush_if_due(now_monotonic=2.0)
    assert result.status == "enqueue_failed"
    assert result.error_id == "background_ingest_enqueue_failed"
    assert "message_bus" not in (result.model_dump_json())
    assert runtime.pending_change_count == 1


def test_runtime_boundary_contract_static() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "file_watcher" / "runtime.py"
    ).read_text(encoding="utf-8")
    for required in (
        "enqueue_background_ingest_job",
        "build_file_watcher_ingest_job",
        "build_incremental_file_change_batch",
        "detect_file_changes",
        "snapshot_allowed_roots",
    ):
        assert required in source
    for forbidden in (
        "message_bus_enqueue",
        "TaskRequest(",
        "TaskHandle(",
        "Kafka",
        "Redis",
        "watchdog",
        "watchfiles",
        "threading",
        "asyncio",
        "time.sleep",
        "while True",
        "Proof" + "Receipt",
        "Mongo" + "Client",
        "py" + "mongo",
    ):
        assert forbidden not in source


def test_config_validation_rejects_bad_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _config(tmp_path, tenant_id=" ")
    with pytest.raises(ValidationError):
        _config(tmp_path, allowed_roots=frozenset())
    with pytest.raises(ValidationError):
        _config(tmp_path, allowed_roots=frozenset({"relative/path"}))
    with pytest.raises(ValidationError):
        _config(tmp_path, debounce_seconds=0)
    with pytest.raises(ValidationError):
        _config(tmp_path, debounce_seconds=2.0, max_batch_wait_seconds=1.0)


def test_protocols_are_exported() -> None:
    assert FileSnapshotProvider is not None
    assert BackgroundIngestEnqueuer is not None


def test_batch_builder_used_for_pending_coalesce(tmp_path: Path) -> None:
    path = _abs(tmp_path, "a.txt")
    created = FileChange(
        kind="created",
        path=str(path),
        current=_snap(path, size=1, mtime=1),
    )
    deleted = FileChange(
        kind="deleted",
        path=str(path),
        previous=_snap(path, size=1, mtime=1),
    )
    batch = build_incremental_file_change_batch((created, deleted))
    assert batch.source_snapshots == ()
    assert batch.deleted_paths == (str(path),)
