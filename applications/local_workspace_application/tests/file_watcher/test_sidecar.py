# © Artur Czarnecki. All rights reserved.

"""Tests for LKW.7B2B file-watcher sidecar process and composition."""

from __future__ import annotations

import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.shutdown import SystemMonotonicClock
from intergrax.hosting.signals import PortableForegroundSignalAdapter
from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueOutput
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.file_watcher import (
    FileSnapshot,
    FileWatcherCheckpoint,
    FileWatcherRuntime,
    FileWatcherRuntimeConfig,
    FileWatcherSidecar,
    FileWatcherSidecarConfig,
    FileWatcherSidecarResult,
    SystemFileWatcherSleeper,
    build_file_watcher_checkpoint,
    build_file_watcher_runtime,
    build_local_workspace_file_watcher_sidecar,
    run_local_workspace_file_watcher_sidecar,
)
from local_workspace_application.file_watcher.contracts import FileChange
from local_workspace_application.file_watcher.runtime import FileWatcherCycleResult
from local_workspace_application.file_watcher.sidecar import _UtcWallClock
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_IDEMPOTENCY_KEY = "lkw.background_ingest.v1:0123456789abcdef0123456789abcdef"


def _snap(path: Path, *, size: int, mtime: int) -> FileSnapshot:
    return FileSnapshot(path=str(path), size_bytes=size, modified_time_ns=mtime)


def _runtime_config(tmp_path: Path, **overrides: object) -> FileWatcherRuntimeConfig:
    payload: dict[str, object] = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "allowed_roots": frozenset({str(tmp_path.resolve())}),
        "debounce_seconds": 1.0,
        "max_batch_wait_seconds": 10.0,
        "priority": "normal",
    }
    payload.update(overrides)
    return FileWatcherRuntimeConfig.model_validate(payload)


def _sidecar_config(tmp_path: Path) -> FileWatcherSidecarConfig:
    return FileWatcherSidecarConfig(
        runtime_config=_runtime_config(tmp_path),
        poll_interval_seconds=1.0,
        checkpoint_path=(
            tmp_path / "data" / "file_watcher" / "checkpoint.json"
        ).resolve(),
    )


class _FakeMonotonicClock:
    def __init__(self, values: list[float] | None = None) -> None:
        self._values = list(
            values or [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        )
        self._index = 0

    def monotonic(self) -> float:
        if self._index >= len(self._values):
            return self._values[-1] + float(self._index)
        value = self._values[self._index]
        self._index += 1
        return value


class _FakeSleeper:
    def __init__(
        self,
        *,
        control: HostedApplicationControlCoordinator | None = None,
        shutdown_after: int | None = None,
        raise_on: int | None = None,
    ) -> None:
        self.sleeps: list[float] = []
        self._control = control
        self._shutdown_after = shutdown_after
        self._raise_on = raise_on

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self._raise_on is not None and len(self.sleeps) == self._raise_on:
            raise RuntimeError("sleep boom")
        if (
            self._control is not None
            and self._shutdown_after is not None
            and len(self.sleeps) >= self._shutdown_after
        ):
            self._control.request_shutdown("test.stop")


class _FakeCheckpointStore:
    def __init__(
        self,
        *,
        checkpoint: FileWatcherCheckpoint | None = None,
        load_error: BaseException | None = None,
        fail_save_on: int | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._load_error = load_error
        self._fail_save_on = fail_save_on
        self.saved: list[FileWatcherCheckpoint] = []
        self.load_calls = 0
        self.save_calls = 0
        self.events: list[str] = []

    def load(self) -> FileWatcherCheckpoint | None:
        self.load_calls += 1
        self.events.append("load")
        if self._load_error is not None:
            raise self._load_error
        return self._checkpoint

    def save(self, checkpoint: FileWatcherCheckpoint) -> None:
        self.save_calls += 1
        self.events.append("save")
        if self._fail_save_on is not None and self.save_calls == self._fail_save_on:
            raise RuntimeError("checkpoint_write_failed")
        self.saved.append(checkpoint)


class _FakeSignalBridge:
    def __init__(
        self,
        *,
        fail_install: bool = False,
        fail_restore: bool = False,
    ) -> None:
        self.install_count = 0
        self.restore_count = 0
        self._fail_install = fail_install
        self._fail_restore = fail_restore

    def install(self) -> None:
        self.install_count += 1
        if self._fail_install:
            raise RuntimeError("install boom")

    def restore(self) -> None:
        self.restore_count += 1
        if self._fail_restore:
            raise RuntimeError("restore boom")


class _FakeSnapshotProvider:
    def __init__(
        self,
        sequence: list[tuple[FileSnapshot, ...] | BaseException],
    ) -> None:
        self._sequence = list(sequence)
        self.calls = 0
        self.events: list[str] = []

    def __call__(self, allowed_roots: frozenset[str]) -> tuple[FileSnapshot, ...]:
        self.calls += 1
        self.events.append("snapshot")
        if not self._sequence:
            raise AssertionError("unexpected snapshot call")
        item = self._sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeEnqueuer:
    def __call__(self, job: object) -> MessageBusEnqueueOutput:
        return MessageBusEnqueueOutput(
            task_id="task-1",
            provider="fake",
            tenant_id="tenant-a",
        )


@dataclass
class _ScriptedRuntime:
    """Minimal runtime double for cycle-status parameterization."""

    cycle_results: list[FileWatcherCycleResult | BaseException]
    checkpoint: FileWatcherCheckpoint
    initialized: bool = False
    initialize_calls: int = 0
    poll_calls: int = 0
    export_calls: int = 0
    restore_calls: int = 0
    events: list[str] = field(default_factory=list)
    initialize_error: BaseException | None = None

    def initialize(self) -> tuple[FileSnapshot, ...]:
        self.initialize_calls += 1
        self.events.append("initialize")
        if self.initialize_error is not None:
            raise self.initialize_error
        self.initialized = True
        return ()

    def restore_checkpoint(
        self,
        checkpoint: FileWatcherCheckpoint,
        *,
        now_monotonic: float,
    ) -> None:
        del now_monotonic
        self.restore_calls += 1
        self.events.append("restore")
        self.checkpoint = checkpoint
        self.initialized = True

    def export_checkpoint(self) -> FileWatcherCheckpoint:
        self.export_calls += 1
        self.events.append("export")
        return self.checkpoint

    def poll_once(self, *, now_monotonic: float) -> FileWatcherCycleResult:
        del now_monotonic
        self.poll_calls += 1
        self.events.append("poll")
        if not self.cycle_results:
            raise AssertionError("unexpected poll_once call")
        item = self.cycle_results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeMessageBus:
    pass


class _RecordingLogger(logging.Logger):
    def __init__(self) -> None:
        super().__init__("test.file_watcher")
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.records.append(("info", str(msg), dict(kwargs.get("extra") or {})))

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.records.append(("warning", str(msg), dict(kwargs.get("extra") or {})))

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.records.append(("error", str(msg), dict(kwargs.get("extra") or {})))


def _enabled_settings(
    tmp_path: Path, **overrides: object
) -> LocalWorkspaceBackendSettings:
    payload: dict[str, object] = {
        "file_watcher_enabled": True,
        "file_watcher_tenant_id": "tenant-a",
        "file_watcher_workspace_id": "workspace-a",
        "file_watcher_collection_id": "collection-a",
        "allowed_read_roots": frozenset({str(tmp_path.resolve())}),
        "data_home": str((tmp_path / "data-home").resolve()),
        "file_watcher_poll_interval_seconds": 1.0,
        "file_watcher_debounce_seconds": 1.0,
        "file_watcher_max_batch_wait_seconds": 10.0,
        "file_watcher_priority": "normal",
    }
    payload.update(overrides)
    return LocalWorkspaceBackendSettings(**payload)  # type: ignore[arg-type]


def _build_sidecar(
    tmp_path: Path,
    *,
    runtime: FileWatcherRuntime | _ScriptedRuntime,
    store: _FakeCheckpointStore,
    sleeper: _FakeSleeper,
    signal_bridge: _FakeSignalBridge | None = None,
    control: HostedApplicationControlCoordinator | None = None,
    clock: _FakeMonotonicClock | None = None,
    logger: logging.Logger | None = None,
) -> tuple[FileWatcherSidecar, HostedApplicationControlCoordinator, _FakeSignalBridge]:
    resolved_control = control or HostedApplicationControlCoordinator(
        clock=_UtcWallClock()
    )
    bridge = signal_bridge or _FakeSignalBridge()
    sidecar = FileWatcherSidecar(
        config=_sidecar_config(tmp_path),
        runtime=runtime,  # type: ignore[arg-type]
        checkpoint_store=store,
        control=resolved_control,
        signal_bridge=bridge,
        monotonic_clock=clock or _FakeMonotonicClock(),
        sleeper=sleeper,
        logger=logger or _RecordingLogger(),
    )
    return sidecar, resolved_control, bridge


def _empty_checkpoint(tmp_path: Path) -> FileWatcherCheckpoint:
    return build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(),
        pending_changes=(),
    )


def test_fresh_start_saves_baseline_polls_before_sleep(tmp_path: Path) -> None:
    provider = _FakeSnapshotProvider([(), ()])
    runtime = FileWatcherRuntime(
        config=_runtime_config(tmp_path),
        snapshot_provider=provider,
        enqueuer=_FakeEnqueuer(),
    )
    store = _FakeCheckpointStore()
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
    )

    result = sidecar.run()

    assert result.exit_kind == "clean_stop"
    assert result.exit_code == 0
    assert result.final_checkpoint_saved is True
    assert result.restored_from_checkpoint is False
    assert runtime.initialized is True
    assert store.load_calls == 1
    assert store.save_calls >= 2
    assert provider.calls >= 2
    assert sleeper.sleeps == [1.0]
    assert bridge.install_count == 1
    assert bridge.restore_count == 1
    # first save is baseline before poll; events: load, save(baseline), then poll path saves
    assert store.events[0] == "load"
    assert store.events[1] == "save"
    assert provider.events[0] == "snapshot"  # initialize
    assert provider.events[1] == "snapshot"  # first poll before sleep


def test_restore_before_poll_detects_downtime_change(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    path = root / "doc.txt"
    baseline = _snap(path, size=1, mtime=100)
    current = _snap(path, size=2, mtime=200)
    checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(root)}),
        baseline_snapshots=(baseline,),
        pending_changes=(),
    )
    provider = _FakeSnapshotProvider([(current,)])
    runtime = FileWatcherRuntime(
        config=_runtime_config(
            tmp_path, debounce_seconds=10.0, max_batch_wait_seconds=30.0
        ),
        snapshot_provider=provider,
        enqueuer=_FakeEnqueuer(),
    )
    store = _FakeCheckpointStore(checkpoint=checkpoint)
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)
    event_log: list[str] = []

    original_load = store.load
    original_poll = runtime.poll_once
    original_save = store.save
    original_sleep = sleeper.sleep

    def load() -> FileWatcherCheckpoint | None:
        event_log.append("load")
        return original_load()

    def poll_once(*, now_monotonic: float) -> FileWatcherCycleResult:
        event_log.append("poll")
        return original_poll(now_monotonic=now_monotonic)

    def save(checkpoint_value: FileWatcherCheckpoint) -> None:
        event_log.append("save")
        return original_save(checkpoint_value)

    def sleep(seconds: float) -> None:
        event_log.append("sleep")
        return original_sleep(seconds)

    store.load = load  # type: ignore[method-assign]
    runtime.poll_once = poll_once  # type: ignore[method-assign]
    store.save = save  # type: ignore[method-assign]
    sleeper.sleep = sleep  # type: ignore[method-assign]

    sidecar, _, _ = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
        clock=_FakeMonotonicClock([10.0, 11.0, 12.0]),
    )

    result = sidecar.run()

    assert result.exit_kind == "clean_stop"
    assert result.restored_from_checkpoint is True
    assert runtime.initialized is True
    assert event_log[:4] == ["load", "poll", "save", "sleep"]
    assert result.last_cycle_status in {"pending", "enqueued"}
    assert result.cycles_completed == 1


def test_invalid_checkpoint_fails_closed(tmp_path: Path) -> None:
    provider = _FakeSnapshotProvider([])
    runtime = FileWatcherRuntime(
        config=_runtime_config(tmp_path),
        snapshot_provider=provider,
        enqueuer=_FakeEnqueuer(),
    )
    store = _FakeCheckpointStore(load_error=RuntimeError("checkpoint_invalid"))
    sleeper = _FakeSleeper()
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "startup_failed"
    assert result.error_id == "checkpoint_restore_failed"
    assert runtime.initialized is False
    assert provider.calls == 0
    assert sleeper.sleeps == []
    assert bridge.restore_count == 1


def test_initial_snapshot_failure(tmp_path: Path) -> None:
    provider = _FakeSnapshotProvider([RuntimeError("file_snapshot_failed")])
    runtime = FileWatcherRuntime(
        config=_runtime_config(tmp_path),
        snapshot_provider=provider,
        enqueuer=_FakeEnqueuer(),
    )
    store = _FakeCheckpointStore()
    sleeper = _FakeSleeper()
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "startup_failed"
    assert result.error_id == "initial_snapshot_failed"
    assert store.save_calls == 0
    assert sleeper.sleeps == []
    assert bridge.restore_count == 1


def test_initial_checkpoint_save_failure(tmp_path: Path) -> None:
    provider = _FakeSnapshotProvider([()])
    runtime = FileWatcherRuntime(
        config=_runtime_config(tmp_path),
        snapshot_provider=provider,
        enqueuer=_FakeEnqueuer(),
    )
    store = _FakeCheckpointStore(fail_save_on=1)
    sleeper = _FakeSleeper()
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "checkpoint_failed"
    assert result.error_id == "checkpoint_write_failed"
    assert result.cycles_completed == 0
    assert provider.calls == 1
    assert sleeper.sleeps == []
    assert bridge.restore_count == 1


def test_immediate_first_poll_before_sleep(tmp_path: Path) -> None:
    events: list[str] = []
    checkpoint = _empty_checkpoint(tmp_path)
    runtime = _ScriptedRuntime(
        cycle_results=[
            FileWatcherCycleResult(status="idle"),
        ],
        checkpoint=checkpoint,
    )
    store = _FakeCheckpointStore()
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)

    original_poll = runtime.poll_once
    original_sleep = sleeper.sleep

    def poll_once(*, now_monotonic: float) -> FileWatcherCycleResult:
        events.append("poll")
        return original_poll(now_monotonic=now_monotonic)

    def sleep(seconds: float) -> None:
        events.append("sleep")
        return original_sleep(seconds)

    runtime.poll_once = poll_once  # type: ignore[method-assign]
    sleeper.sleep = sleep  # type: ignore[method-assign]

    sidecar, _, _ = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
    )
    result = sidecar.run()

    assert result.exit_kind == "clean_stop"
    assert events[:2] == ["poll", "sleep"]
    assert "sleep" not in events[:1]


@pytest.mark.parametrize(
    "status",
    ["idle", "pending", "enqueued", "deletions_only", "enqueue_failed"],
)
def test_save_after_every_cycle_status(tmp_path: Path, status: str) -> None:
    if status == "idle":
        cycle = FileWatcherCycleResult(status="idle")
    elif status == "pending":
        cycle = FileWatcherCycleResult(status="pending", pending_change_count=1)
    elif status == "enqueued":
        cycle = FileWatcherCycleResult(
            status="enqueued",
            actionable_path_count=1,
            change_token="sha256:" + ("a" * 64),
            task_id=_IDEMPOTENCY_KEY,
            provider="fake",
            tenant_id="tenant-a",
            broker_run_id=_IDEMPOTENCY_KEY,
            idempotency_key=_IDEMPOTENCY_KEY,
        )
    elif status == "deletions_only":
        cycle = FileWatcherCycleResult(
            status="deletions_only",
            deleted_path_count=1,
        )
    else:
        cycle = FileWatcherCycleResult(
            status="enqueue_failed",
            pending_change_count=1,
            actionable_path_count=1,
            change_token="sha256:" + ("b" * 64),
            error_id="background_ingest_enqueue_failed",
        )

    pending_checkpoint = build_file_watcher_checkpoint(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
        baseline_snapshots=(),
        pending_changes=(),
    )
    if status in {"pending", "enqueue_failed"}:
        path = tmp_path.resolve() / "x.txt"
        snap = _snap(path, size=1, mtime=1)
        change = FileChange(kind="created", path=str(path), current=snap)
        pending_checkpoint = build_file_watcher_checkpoint(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            allowed_roots=frozenset({str(tmp_path.resolve())}),
            baseline_snapshots=(snap,),
            pending_changes=(change,),
        )

    runtime = _ScriptedRuntime(
        cycle_results=[cycle],
        checkpoint=pending_checkpoint,
    )
    store = _FakeCheckpointStore()
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)
    sidecar, _, _ = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
    )

    result = sidecar.run()

    assert result.exit_kind == "clean_stop"
    assert result.cycles_completed == 1
    assert result.last_cycle_status == status
    # baseline save + cycle save + final save
    assert store.save_calls >= 2
    assert "save" in store.events
    if status == "enqueue_failed":
        assert result.final_checkpoint_saved is True
        assert store.saved[-1] == pending_checkpoint


def test_snapshot_failure_retries_without_checkpoint(tmp_path: Path) -> None:
    logger = _RecordingLogger()
    runtime = _ScriptedRuntime(
        cycle_results=[
            RuntimeError("file_snapshot_failed"),
            FileWatcherCycleResult(status="idle"),
        ],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore()
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=2)
    sidecar, _, _ = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
        logger=logger,
    )

    result = sidecar.run()

    assert result.exit_kind == "clean_stop"
    assert result.cycles_completed == 1
    assert sleeper.sleeps == [1.0, 1.0]
    assert runtime.poll_calls == 2
    # baseline + success cycle + final (no save after snapshot failure)
    assert store.save_calls == 3
    error_logs = [item for item in logger.records if item[0] == "error"]
    assert error_logs
    assert error_logs[0][2].get("error_id") == "file_snapshot_failed"
    serialized = str(logger.records)
    assert "doc.txt" not in serialized
    assert str(tmp_path) not in serialized


def test_unknown_runtime_failure(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[RuntimeError("unexpected")],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore()
    sleeper = _FakeSleeper()
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "runtime_failed"
    assert result.error_id == "file_watcher_runtime_failed"
    assert "unexpected" not in (result.error_id or "")
    assert "unexpected" not in result.model_dump_json()
    assert sleeper.sleeps == []
    assert bridge.restore_count == 1


def test_checkpoint_failure_after_pending(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[
            FileWatcherCycleResult(status="pending", pending_change_count=1),
        ],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    # fail_save_on=2: first save is fresh baseline; second is after cycle
    store = _FakeCheckpointStore(fail_save_on=2)
    sleeper = _FakeSleeper()
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "checkpoint_failed"
    assert result.cycles_completed == 1
    assert sleeper.sleeps == []
    assert bridge.restore_count == 1


def test_checkpoint_failure_after_enqueue_success(tmp_path: Path) -> None:
    """Restart safety relies on existing deterministic background-ingest idempotency key."""
    runtime = _ScriptedRuntime(
        cycle_results=[
            FileWatcherCycleResult(
                status="enqueued",
                actionable_path_count=1,
                change_token="sha256:" + ("c" * 64),
                task_id=_IDEMPOTENCY_KEY,
                provider="fake",
                tenant_id="tenant-a",
                broker_run_id=_IDEMPOTENCY_KEY,
                idempotency_key=_IDEMPOTENCY_KEY,
            ),
        ],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore(fail_save_on=2)
    sleeper = _FakeSleeper()
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "checkpoint_failed"
    assert result.exit_code == 1
    assert result.cycles_completed == 1
    assert sleeper.sleeps == []
    assert runtime.poll_calls == 1
    assert bridge.restore_count == 1


def test_graceful_shutdown_saves_final_checkpoint(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[FileWatcherCycleResult(status="idle")],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore()
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
    )

    result = sidecar.run()

    assert result.exit_kind == "clean_stop"
    assert result.final_checkpoint_saved is True
    assert store.save_calls >= 2
    assert bridge.restore_count == 1


def test_final_checkpoint_failure(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[FileWatcherCycleResult(status="idle")],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    # saves: baseline(1), after cycle(2), final(3)
    store = _FakeCheckpointStore(fail_save_on=3)
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
    )

    result = sidecar.run()

    assert result.exit_kind == "checkpoint_failed"
    assert result.final_checkpoint_saved is False
    assert result.exit_code == 1
    assert bridge.restore_count == 1


def test_signal_install_failure(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore()
    bridge = _FakeSignalBridge(fail_install=True)
    sleeper = _FakeSleeper()
    sidecar, _, _ = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        signal_bridge=bridge,
    )

    result = sidecar.run()

    assert result.exit_kind == "startup_failed"
    assert result.error_id == "signal_install_failed"
    assert bridge.restore_count == 0
    assert store.load_calls == 0
    assert runtime.initialize_calls == 0


def test_signal_restore_failure(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[FileWatcherCycleResult(status="idle")],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore()
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    sleeper = _FakeSleeper(control=control, shutdown_after=1)
    bridge = _FakeSignalBridge(fail_restore=True)
    sidecar, _, _ = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
        control=control,
        signal_bridge=bridge,
    )

    result = sidecar.run()

    assert result.exit_kind == "runtime_failed"
    assert result.error_id == "signal_restore_failed"
    assert result.exit_code == 1


def test_sleeper_failure(tmp_path: Path) -> None:
    runtime = _ScriptedRuntime(
        cycle_results=[FileWatcherCycleResult(status="idle")],
        checkpoint=_empty_checkpoint(tmp_path),
    )
    store = _FakeCheckpointStore()
    sleeper = _FakeSleeper(raise_on=1)
    sidecar, _, bridge = _build_sidecar(
        tmp_path,
        runtime=runtime,
        store=store,
        sleeper=sleeper,
    )

    result = sidecar.run()

    assert result.exit_kind == "runtime_failed"
    assert result.error_id == "file_watcher_sleep_failed"
    assert bridge.restore_count == 1


@pytest.mark.parametrize(
    "result",
    [
        FileWatcherSidecarResult(
            exit_kind="clean_stop",
            exit_code=0,
            final_checkpoint_saved=True,
        ),
        FileWatcherSidecarResult(
            exit_kind="disabled",
            exit_code=2,
            error_id="file_watcher_disabled",
        ),
        FileWatcherSidecarResult(
            exit_kind="configuration_error",
            exit_code=2,
            error_id="message_bus_not_enabled",
        ),
        FileWatcherSidecarResult(
            exit_kind="startup_failed",
            exit_code=1,
            error_id="signal_install_failed",
        ),
        FileWatcherSidecarResult(
            exit_kind="checkpoint_failed",
            exit_code=1,
            error_id="checkpoint_write_failed",
        ),
        FileWatcherSidecarResult(
            exit_kind="runtime_failed",
            exit_code=1,
            error_id="file_watcher_runtime_failed",
        ),
    ],
)
def test_safe_result_serialization(result: FileWatcherSidecarResult) -> None:
    payload = result.model_dump(mode="json")
    text = result.model_dump_json()
    for token in (
        "path",
        "tenant_id",
        "workspace_id",
        "collection_id",
        "change_token",
        "task_id",
        "provider",
        "payload",
        "exception",
        "Kafka",
        "Redis",
        "credentials",
        "broker_url",
    ):
        assert token not in payload
        assert f'"{token}"' not in text


def test_safe_result_has_no_sensitive_keys() -> None:
    result = FileWatcherSidecarResult(
        exit_kind="clean_stop",
        exit_code=0,
        final_checkpoint_saved=True,
        last_cycle_status="enqueued",
        cycles_completed=3,
        restored_from_checkpoint=True,
    )
    keys = set(result.model_dump(mode="json"))
    for token in (
        "path",
        "tenant_id",
        "workspace_id",
        "collection_id",
        "change_token",
        "task_id",
        "provider",
        "payload",
        "exception",
        "Kafka",
        "Redis",
        "credentials",
        "broker_url",
    ):
        assert token not in keys


class _FakeSignalApi:
    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {}
        self.previous: dict[int, Any] = {}

    def getsignal(self, signum: int) -> Any:
        return self.previous.get(signum, signal.SIG_DFL)

    def signal(self, signum: int, handler: Any) -> Any:
        previous = self.handlers.get(signum, signal.SIG_DFL)
        self.handlers[signum] = handler
        return previous


def test_production_signal_adapter_composition() -> None:
    control = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    api = _FakeSignalApi()
    adapter = PortableForegroundSignalAdapter(
        coordinator=control,
        signal_api=api,
        enable_sighup_restart=False,
    )
    adapter.install()
    assert signal.SIGINT in api.handlers
    assert signal.SIGTERM in api.handlers
    if hasattr(signal, "SIGBREAK"):
        assert signal.SIGBREAK in api.handlers

    api.handlers[signal.SIGINT](signal.SIGINT, None)
    assert control.is_shutdown_requested()

    control2 = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    api2 = _FakeSignalApi()
    adapter2 = PortableForegroundSignalAdapter(
        coordinator=control2,
        signal_api=api2,
        enable_sighup_restart=False,
    )
    adapter2.install()
    api2.handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert control2.is_shutdown_requested()

    previous = object()
    api3 = _FakeSignalApi()
    api3.previous[signal.SIGINT] = previous
    control3 = HostedApplicationControlCoordinator(clock=_UtcWallClock())
    adapter3 = PortableForegroundSignalAdapter(
        coordinator=control3,
        signal_api=api3,
        enable_sighup_restart=False,
    )
    adapter3.install()
    adapter3.restore()
    assert api3.handlers[signal.SIGINT] is previous


def test_disabled_composition_skips_kafka(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = {"kafka": False}

    def _boom() -> object:
        called["kafka"] = True
        raise AssertionError("kafka must not be created")

    monkeypatch.setattr(
        "local_workspace_application.file_watcher.sidecar.create_local_workspace_kafka_message_bus",
        _boom,
    )
    settings = _enabled_settings(tmp_path, file_watcher_enabled=False)
    result = run_local_workspace_file_watcher_sidecar(settings=settings)
    assert result.exit_kind == "disabled"
    assert result.exit_code == 2
    assert called["kafka"] is False


def test_message_bus_not_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = {"kafka": False}

    def _boom() -> object:
        called["kafka"] = True
        raise AssertionError("kafka must not be created")

    monkeypatch.setattr(
        "local_workspace_application.file_watcher.sidecar.local_workspace_message_bus_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "local_workspace_application.file_watcher.sidecar.create_local_workspace_kafka_message_bus",
        _boom,
    )
    settings = _enabled_settings(tmp_path)
    result = run_local_workspace_file_watcher_sidecar(settings=settings)
    assert result.exit_kind == "configuration_error"
    assert result.error_id == "message_bus_not_enabled"
    assert called["kafka"] is False


def test_message_bus_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "local_workspace_application.file_watcher.sidecar.local_workspace_message_bus_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "local_workspace_application.file_watcher.sidecar.create_local_workspace_kafka_message_bus",
        lambda: (_ for _ in ()).throw(
            RuntimeError("broker localhost:9092 redis://secret")
        ),
    )
    settings = _enabled_settings(tmp_path)
    result = run_local_workspace_file_watcher_sidecar(settings=settings)
    assert result.exit_kind == "startup_failed"
    assert result.error_id == "message_bus_initialization_failed"
    assert "9092" not in result.model_dump_json()
    assert "redis" not in result.model_dump_json().lower()


def test_production_builder_uses_existing_boundaries(tmp_path: Path) -> None:
    settings = _enabled_settings(tmp_path)
    bus = _FakeMessageBus()
    sidecar = build_local_workspace_file_watcher_sidecar(
        settings=settings,
        message_bus=bus,  # type: ignore[arg-type]
        working_directory=tmp_path,
    )
    assert isinstance(sidecar._runtime, FileWatcherRuntime)
    assert isinstance(sidecar._monotonic_clock, SystemMonotonicClock)
    assert isinstance(sidecar._sleeper, SystemFileWatcherSleeper)
    assert isinstance(sidecar._signal_bridge, PortableForegroundSignalAdapter)
    assert isinstance(sidecar._control, HostedApplicationControlCoordinator)
    expected = (
        Path(settings.data_home).resolve() / "data" / "file_watcher" / "checkpoint.json"
    )
    assert sidecar._config.checkpoint_path == expected
    # ToolWiringContext.message_bus is the injected bus
    wiring = ToolWiringContext(message_bus=bus)  # type: ignore[arg-type]
    assert wiring.message_bus is bus
    # rebuild through helper to confirm build_file_watcher_runtime path
    runtime = build_file_watcher_runtime(
        config=sidecar._config.runtime_config,
        wiring_context=ToolWiringContext(message_bus=bus),  # type: ignore[arg-type]
    )
    assert isinstance(runtime, FileWatcherRuntime)


def test_boundary_static_inspection() -> None:
    sidecar_path = Path(__file__).resolve().parents[2] / "file_watcher" / "sidecar.py"
    main_path = Path(__file__).resolve().parents[2] / "file_watcher" / "__main__.py"
    sidecar_text = sidecar_path.read_text(encoding="utf-8")
    main_text = main_path.read_text(encoding="utf-8")
    combined = sidecar_text + "\n" + main_text

    for required in (
        "PortableForegroundSignalAdapter",
        "HostedApplicationControlCoordinator",
        "SystemMonotonicClock",
        "restore_file_watcher_runtime",
        "build_file_watcher_runtime",
        "JsonFileWatcherCheckpointStore",
        "create_local_workspace_kafka_message_bus",
    ):
        assert required in sidecar_text

    for forbidden in (
        "message_bus_enqueue",
        "TaskRequest(",
        "TaskHandle(",
        "create_kafka_message_bus",
        "create_redis_kv_store",
        "watchdog",
        "watchfiles",
        "inotify",
        "ReadDirectoryChangesW",
        "threading.Thread",
        "asyncio.create_task",
        "signal.signal(",
        "ProofReceipt",
        "MongoClient",
        "pymongo",
    ):
        assert forbidden not in combined

    # time.sleep only inside SystemFileWatcherSleeper.sleep
    run_method = sidecar_text.split("def run(self)", 1)[1].split(
        "def _run_after_signals", 1
    )[0]
    assert "time.sleep" not in run_method
    assert "time.sleep(seconds)" in sidecar_text
