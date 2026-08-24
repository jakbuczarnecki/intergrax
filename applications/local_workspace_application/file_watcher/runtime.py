# © Artur Czarnecki. All rights reserved.

"""Deterministic file-watcher runtime state machine (LKW.7B1/7B2A).

Owns one poll cycle, pending coalescing, bounded debounce, enqueue through
enqueue_background_ingest_job(), and durable checkpoint export/restore.
No OS process, sleep, or loop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueOutput
from intergrax.tools.registry.wiring import ToolWiringContext

from local_workspace_application.background_ingest.contracts import (
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
)
from local_workspace_application.background_ingest.enqueue import (
    enqueue_background_ingest_job,
)
from local_workspace_application.file_watcher.batching import (
    build_file_watcher_ingest_job,
    build_incremental_file_change_batch,
)
from local_workspace_application.file_watcher.checkpoint import (
    FileWatcherCheckpoint,
    build_file_watcher_checkpoint,
)
from local_workspace_application.file_watcher.contracts import (
    FileChange,
    FileSnapshot,
    normalize_watch_path_key,
)
from local_workspace_application.file_watcher.snapshot import (
    detect_file_changes,
    snapshot_allowed_roots,
)

FileWatcherCycleStatus = Literal[
    "idle",
    "pending",
    "enqueued",
    "deletions_only",
    "enqueue_failed",
]

_ENQUEUE_FAILED_ERROR_ID = "background_ingest_enqueue_failed"


class FileSnapshotProvider(Protocol):
    def __call__(
        self,
        allowed_roots: frozenset[str],
    ) -> tuple[FileSnapshot, ...]: ...


class BackgroundIngestEnqueuer(Protocol):
    def __call__(
        self,
        job: LkwBackgroundIngestJob,
    ) -> MessageBusEnqueueOutput: ...


class FileWatcherRuntimeConfig(BaseModel):
    """Immutable watcher runtime configuration (LKW.7B1)."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    workspace_id: str
    collection_id: str
    allowed_roots: frozenset[str]

    debounce_seconds: float = 1.0
    max_batch_wait_seconds: float = 10.0
    priority: str = "normal"

    @field_validator("tenant_id", "workspace_id", "collection_id", "priority")
    @classmethod
    def _require_non_blank(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-blank string")
        return value

    @field_validator("allowed_roots")
    @classmethod
    def _validate_allowed_roots(cls, value: object) -> frozenset[str]:
        if not isinstance(value, (set, frozenset)):
            raise ValueError("allowed_roots must be a frozenset")
        roots = frozenset(str(item) for item in value)
        if not roots:
            raise ValueError("allowed_roots must be non-empty")
        for root in roots:
            if not root.strip():
                raise ValueError("allowed root must be non-blank")
            if not Path(root).expanduser().is_absolute():
                raise ValueError("allowed root must be absolute")
        return roots

    @field_validator("debounce_seconds", "max_batch_wait_seconds")
    @classmethod
    def _validate_positive_finite(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a finite number greater than 0")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("must be a finite number greater than 0")
        return number

    @model_validator(mode="after")
    def _validate_wait_bounds(self) -> FileWatcherRuntimeConfig:
        if self.max_batch_wait_seconds < self.debounce_seconds:
            raise ValueError(
                "max_batch_wait_seconds must be greater than or equal to debounce_seconds"
            )
        return self


class FileWatcherCycleResult(BaseModel):
    """Safe structured result of one watcher cycle (no paths or payloads)."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["lkw.file_watcher_cycle_result.v1"] = (
        "lkw.file_watcher_cycle_result.v1"
    )

    status: FileWatcherCycleStatus

    detected_change_count: int = Field(default=0, ge=0)
    pending_change_count: int = Field(default=0, ge=0)
    actionable_path_count: int = Field(default=0, ge=0)
    deleted_path_count: int = Field(default=0, ge=0)

    change_token: str | None = None

    task_id: str | None = None
    provider: str | None = None
    tenant_id: str | None = None
    broker_run_id: str | None = None
    idempotency_key: str | None = None

    error_id: str | None = None

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> FileWatcherCycleResult:
        if self.status == "idle":
            if self.pending_change_count != 0:
                raise ValueError("idle requires pending_change_count == 0")
            if self.task_id is not None or self.provider is not None:
                raise ValueError("idle must not include task_id or provider")
            if self.error_id is not None:
                raise ValueError("idle must not include error_id")
            return self
        if self.status == "pending":
            if self.pending_change_count <= 0:
                raise ValueError("pending requires pending_change_count > 0")
            if self.task_id is not None or self.provider is not None:
                raise ValueError("pending must not include task_id or provider")
            if self.error_id is not None:
                raise ValueError("pending must not include error_id")
            return self
        if self.status == "enqueued":
            if self.pending_change_count != 0:
                raise ValueError("enqueued requires pending_change_count == 0")
            if self.actionable_path_count <= 0:
                raise ValueError("enqueued requires actionable_path_count > 0")
            if not self.change_token or not self.change_token.strip():
                raise ValueError("enqueued requires a non-empty change_token")
            if not self.task_id or not self.task_id.strip():
                raise ValueError("enqueued requires a non-empty task_id")
            if not self.provider or not self.provider.strip():
                raise ValueError("enqueued requires a non-empty provider")
            if not self.broker_run_id or not self.broker_run_id.strip():
                raise ValueError("enqueued requires a non-empty broker_run_id")
            if not self.idempotency_key or not self.idempotency_key.strip():
                raise ValueError("enqueued requires a non-empty idempotency_key")
            if self.broker_run_id != self.idempotency_key:
                raise ValueError("enqueued requires broker_run_id == idempotency_key")
            if self.task_id != self.broker_run_id:
                raise ValueError("enqueued requires task_id == broker_run_id")
            if self.error_id is not None:
                raise ValueError("enqueued must not include error_id")
            return self
        if self.status == "deletions_only":
            if self.pending_change_count != 0:
                raise ValueError("deletions_only requires pending_change_count == 0")
            if self.actionable_path_count != 0:
                raise ValueError("deletions_only requires actionable_path_count == 0")
            if self.deleted_path_count <= 0:
                raise ValueError("deletions_only requires deleted_path_count > 0")
            if self.change_token is not None:
                raise ValueError("deletions_only requires change_token is None")
            if self.task_id is not None or self.provider is not None:
                raise ValueError("deletions_only must not include task_id or provider")
            if self.error_id is not None:
                raise ValueError("deletions_only must not include error_id")
            return self
        if self.status == "enqueue_failed":
            if self.pending_change_count <= 0:
                raise ValueError("enqueue_failed requires pending_change_count > 0")
            if self.actionable_path_count <= 0:
                raise ValueError("enqueue_failed requires actionable_path_count > 0")
            if not self.change_token or not self.change_token.strip():
                raise ValueError("enqueue_failed requires a non-empty change_token")
            if self.task_id is not None or self.provider is not None:
                raise ValueError("enqueue_failed must not include task_id or provider")
            if self.error_id != _ENQUEUE_FAILED_ERROR_ID:
                raise ValueError(
                    f"enqueue_failed requires error_id == {_ENQUEUE_FAILED_ERROR_ID!r}"
                )
            return self
        raise ValueError(f"unsupported cycle status: {self.status}")


class FileWatcherRuntime:
    """Deterministic polling/debounce/enqueue state machine for LKW.7B1."""

    def __init__(
        self,
        *,
        config: FileWatcherRuntimeConfig,
        snapshot_provider: FileSnapshotProvider,
        enqueuer: BackgroundIngestEnqueuer,
    ) -> None:
        self._config = config
        self._snapshot_provider = snapshot_provider
        self._enqueuer = enqueuer
        self._initialized = False
        self._baseline_snapshots: tuple[FileSnapshot, ...] = ()
        self._pending_changes_by_key: dict[str, FileChange] = {}
        self._first_pending_at: float | None = None
        self._last_change_at: float | None = None
        self._last_observed_monotonic: float | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def baseline_file_count(self) -> int:
        return len(self._baseline_snapshots)

    @property
    def pending_change_count(self) -> int:
        return len(self._pending_changes_by_key)

    def initialize(self) -> tuple[FileSnapshot, ...]:
        """Acquire baseline snapshots without emitting created events or enqueue."""
        snapshots = self._snapshot_provider(self._config.allowed_roots)
        self._baseline_snapshots = snapshots
        self._initialized = True
        self._pending_changes_by_key.clear()
        self._first_pending_at = None
        self._last_change_at = None
        self._last_observed_monotonic = None
        return snapshots

    def export_checkpoint(self) -> FileWatcherCheckpoint:
        """Export current baseline and final pending changes as a checkpoint."""
        self._require_initialized()
        ordered_pending = tuple(
            self._pending_changes_by_key[key]
            for key in sorted(self._pending_changes_by_key.keys())
        )
        return build_file_watcher_checkpoint(
            tenant_id=self._config.tenant_id,
            workspace_id=self._config.workspace_id,
            collection_id=self._config.collection_id,
            allowed_roots=self._config.allowed_roots,
            baseline_snapshots=self._baseline_snapshots,
            pending_changes=ordered_pending,
        )

    def restore_checkpoint(
        self,
        checkpoint: FileWatcherCheckpoint,
        *,
        now_monotonic: float,
    ) -> None:
        """Restore baseline and pending state; start a new process clock epoch."""
        accepted = self._validate_restore_monotonic(now_monotonic)
        try:
            validated = FileWatcherCheckpoint.model_validate(
                checkpoint.model_dump(mode="json")
            )
        except ValidationError:
            raise RuntimeError("checkpoint_invalid") from None
        self._validate_checkpoint_identity(validated)

        pending_map = {
            normalize_watch_path_key(change.path): change
            for change in validated.pending_changes
        }

        self._initialized = True
        self._baseline_snapshots = validated.baseline_snapshots
        self._pending_changes_by_key = pending_map
        self._last_observed_monotonic = accepted
        if pending_map:
            self._first_pending_at = accepted
            self._last_change_at = accepted
        else:
            self._first_pending_at = None
            self._last_change_at = None

    def poll_once(self, *, now_monotonic: float) -> FileWatcherCycleResult:
        """Snapshot, diff, merge pending state, then evaluate flush eligibility."""
        self._require_initialized()
        accepted = self._validate_monotonic(now_monotonic)
        current = self._snapshot_provider(self._config.allowed_roots)
        changes = detect_file_changes(self._baseline_snapshots, current)
        self._baseline_snapshots = current
        if changes:
            for change in changes:
                key = normalize_watch_path_key(change.path)
                self._pending_changes_by_key[key] = change
            if self._first_pending_at is None:
                self._first_pending_at = accepted
            self._last_change_at = accepted
        return self._evaluate_flush(
            now_monotonic=accepted,
            detected_change_count=len(changes),
        )

    def flush_if_due(self, *, now_monotonic: float) -> FileWatcherCycleResult:
        """Evaluate pending debounce deadlines without taking a new snapshot."""
        self._require_initialized()
        accepted = self._validate_monotonic(now_monotonic)
        return self._evaluate_flush(
            now_monotonic=accepted,
            detected_change_count=0,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("file_watcher_not_initialized")

    def _validate_restore_monotonic(self, now_monotonic: object) -> float:
        if isinstance(now_monotonic, bool) or not isinstance(
            now_monotonic, (int, float)
        ):
            raise RuntimeError("invalid_monotonic_time")
        value = float(now_monotonic)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("invalid_monotonic_time")
        return value

    def _validate_checkpoint_identity(self, checkpoint: FileWatcherCheckpoint) -> None:
        expected = build_file_watcher_checkpoint(
            tenant_id=self._config.tenant_id,
            workspace_id=self._config.workspace_id,
            collection_id=self._config.collection_id,
            allowed_roots=self._config.allowed_roots,
            baseline_snapshots=(),
            pending_changes=(),
        )
        if (
            checkpoint.tenant_id != expected.tenant_id
            or checkpoint.workspace_id != expected.workspace_id
            or checkpoint.collection_id != expected.collection_id
            or checkpoint.allowed_roots != expected.allowed_roots
        ):
            raise RuntimeError("checkpoint_identity_mismatch")

    def _validate_monotonic(self, now_monotonic: object) -> float:
        if isinstance(now_monotonic, bool) or not isinstance(
            now_monotonic, (int, float)
        ):
            raise RuntimeError("invalid_monotonic_time")
        value = float(now_monotonic)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("invalid_monotonic_time")
        if (
            self._last_observed_monotonic is not None
            and value < self._last_observed_monotonic
        ):
            raise RuntimeError("monotonic_time_regressed")
        return value

    def _is_due(self, now_monotonic: float) -> bool:
        if self._last_change_at is None or self._first_pending_at is None:
            return False
        quiet_elapsed = now_monotonic - self._last_change_at
        wait_elapsed = now_monotonic - self._first_pending_at
        return (
            quiet_elapsed >= self._config.debounce_seconds
            or wait_elapsed >= self._config.max_batch_wait_seconds
        )

    def _clear_pending(self) -> None:
        self._pending_changes_by_key.clear()
        self._first_pending_at = None
        self._last_change_at = None

    def _accept_time(self, now_monotonic: float) -> None:
        self._last_observed_monotonic = now_monotonic

    def _evaluate_flush(
        self,
        *,
        now_monotonic: float,
        detected_change_count: int,
    ) -> FileWatcherCycleResult:
        if not self._pending_changes_by_key:
            result = FileWatcherCycleResult(
                status="idle",
                detected_change_count=detected_change_count,
                pending_change_count=0,
            )
            self._accept_time(now_monotonic)
            return result

        if not self._is_due(now_monotonic):
            result = FileWatcherCycleResult(
                status="pending",
                detected_change_count=detected_change_count,
                pending_change_count=len(self._pending_changes_by_key),
            )
            self._accept_time(now_monotonic)
            return result

        ordered_changes = tuple(
            self._pending_changes_by_key[key]
            for key in sorted(self._pending_changes_by_key.keys())
        )
        batch = build_incremental_file_change_batch(ordered_changes)

        if not batch.source_snapshots:
            deleted_path_count = len(batch.deleted_paths)
            self._clear_pending()
            result = FileWatcherCycleResult(
                status="deletions_only",
                detected_change_count=detected_change_count,
                pending_change_count=0,
                actionable_path_count=0,
                deleted_path_count=deleted_path_count,
                change_token=None,
            )
            self._accept_time(now_monotonic)
            return result

        job = build_file_watcher_ingest_job(
            batch,
            tenant_id=self._config.tenant_id,
            workspace_id=self._config.workspace_id,
            collection_id=self._config.collection_id,
            allowed_roots=self._config.allowed_roots,
            priority=self._config.priority,
        )

        try:
            output = self._enqueuer(job)
        except Exception:
            result = FileWatcherCycleResult(
                status="enqueue_failed",
                detected_change_count=detected_change_count,
                pending_change_count=len(self._pending_changes_by_key),
                actionable_path_count=len(batch.source_snapshots),
                deleted_path_count=len(batch.deleted_paths),
                change_token=batch.change_token,
                error_id=_ENQUEUE_FAILED_ERROR_ID,
            )
            self._accept_time(now_monotonic)
            return result

        if not output.task_id.strip() or not output.provider.strip():
            result = FileWatcherCycleResult(
                status="enqueue_failed",
                detected_change_count=detected_change_count,
                pending_change_count=len(self._pending_changes_by_key),
                actionable_path_count=len(batch.source_snapshots),
                deleted_path_count=len(batch.deleted_paths),
                change_token=batch.change_token,
                error_id=_ENQUEUE_FAILED_ERROR_ID,
            )
            self._accept_time(now_monotonic)
            return result

        resolved_idempotency_key = background_ingest_idempotency_key(job)
        self._clear_pending()
        result = FileWatcherCycleResult(
            status="enqueued",
            detected_change_count=detected_change_count,
            pending_change_count=0,
            actionable_path_count=len(batch.source_snapshots),
            deleted_path_count=len(batch.deleted_paths),
            change_token=batch.change_token,
            task_id=output.task_id,
            provider=output.provider,
            tenant_id=output.tenant_id or self._config.tenant_id,
            broker_run_id=resolved_idempotency_key,
            idempotency_key=resolved_idempotency_key,
        )
        self._accept_time(now_monotonic)
        return result


def build_file_watcher_runtime(
    *,
    config: FileWatcherRuntimeConfig,
    wiring_context: ToolWiringContext,
    snapshot_provider: FileSnapshotProvider = snapshot_allowed_roots,
) -> FileWatcherRuntime:
    """Bind production enqueue through the existing background-ingest helper."""

    def _enqueuer(job: LkwBackgroundIngestJob) -> MessageBusEnqueueOutput:
        return enqueue_background_ingest_job(wiring_context, job)

    return FileWatcherRuntime(
        config=config,
        snapshot_provider=snapshot_provider,
        enqueuer=_enqueuer,
    )
