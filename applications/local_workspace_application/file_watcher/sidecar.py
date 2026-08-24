# © Artur Czarnecki. All rights reserved.

"""Cross-platform file-watcher sidecar process (LKW.7B2B).

Owns settings validation, process loop, signal bridge, and automatic
checkpoint lifecycle. Diff/debounce/batch/enqueue remain in FileWatcherRuntime.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.shutdown import MonotonicClock, SystemMonotonicClock
from intergrax.hosting.signals import (
    HostedApplicationSignalBridge,
    PortableForegroundSignalAdapter,
)
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.file_watcher.checkpoint import (
    FileWatcherCheckpoint,
    JsonFileWatcherCheckpointStore,
    file_watcher_checkpoint_path,
    restore_file_watcher_runtime,
)
from local_workspace_application.file_watcher.execution_evidence import (
    FileWatcherIngestEnqueuedRecord,
)
from local_workspace_application.file_watcher.runtime import (
    FileWatcherCycleStatus,
    FileWatcherRuntime,
    FileWatcherRuntimeConfig,
    build_file_watcher_runtime,
)
from local_workspace_application.host.message_bus_wiring import (
    create_local_workspace_kafka_message_bus,
    local_workspace_message_bus_enabled,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_SNAPSHOT_ERROR_IDS = frozenset(
    {
        "read_allowlist_not_configured",
        "watch_root_not_absolute",
        "watch_root_not_found",
        "watch_root_not_directory",
        "path_not_in_allowlist",
        "file_snapshot_failed",
    }
)

_VALID_PRIORITIES = frozenset({"low", "normal", "high"})

FileWatcherSidecarExitKind = Literal[
    "clean_stop",
    "disabled",
    "configuration_error",
    "startup_failed",
    "checkpoint_failed",
    "runtime_failed",
]


class FileWatcherSidecarConfigurationError(RuntimeError):
    """Stable configuration failure identified only by error ID."""


class FileWatcherSleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...


class SystemFileWatcherSleeper:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FileWatcherCheckpointStore(Protocol):
    def load(self) -> FileWatcherCheckpoint | None: ...

    def save(self, checkpoint: FileWatcherCheckpoint) -> None: ...


class _UtcWallClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FileWatcherSidecarConfig(BaseModel):
    """Immutable sidecar process configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_config: FileWatcherRuntimeConfig
    poll_interval_seconds: float
    checkpoint_path: Path

    @field_validator("poll_interval_seconds")
    @classmethod
    def _validate_poll_interval(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "poll_interval_seconds must be a finite number greater than 0"
            )
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                "poll_interval_seconds must be a finite number greater than 0"
            )
        return number

    @field_validator("checkpoint_path")
    @classmethod
    def _validate_checkpoint_path(cls, value: object) -> Path:
        if not isinstance(value, Path):
            raise ValueError("checkpoint_path must be a Path")
        if not value.is_absolute():
            raise ValueError("checkpoint_path must be absolute")
        return value


class FileWatcherSidecarResult(BaseModel):
    """Safe structured sidecar process result (no paths or payloads)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["lkw.file_watcher_sidecar_result.v1"] = (
        "lkw.file_watcher_sidecar_result.v1"
    )

    exit_kind: FileWatcherSidecarExitKind
    exit_code: int

    restored_from_checkpoint: bool = False
    cycles_completed: int = Field(default=0, ge=0)

    last_cycle_status: FileWatcherCycleStatus | None = None

    final_checkpoint_saved: bool = False
    error_id: str | None = None

    @model_validator(mode="after")
    def _validate_result_invariants(self) -> FileWatcherSidecarResult:
        if self.exit_kind == "clean_stop":
            if self.exit_code != 0:
                raise ValueError("clean_stop requires exit_code == 0")
            if self.error_id is not None:
                raise ValueError("clean_stop requires error_id is None")
            if not self.final_checkpoint_saved:
                raise ValueError("clean_stop requires final_checkpoint_saved is True")
            return self
        if self.exit_kind == "disabled":
            if self.exit_code != 2:
                raise ValueError("disabled requires exit_code == 2")
            if self.error_id != "file_watcher_disabled":
                raise ValueError("disabled requires error_id == file_watcher_disabled")
            if self.cycles_completed != 0:
                raise ValueError("disabled requires cycles_completed == 0")
            return self
        if self.exit_kind == "configuration_error":
            if self.exit_code != 2:
                raise ValueError("configuration_error requires exit_code == 2")
            if not self.error_id or not self.error_id.strip():
                raise ValueError("configuration_error requires a non-empty error_id")
            if self.cycles_completed != 0:
                raise ValueError("configuration_error requires cycles_completed == 0")
            return self
        if self.exit_kind == "startup_failed":
            if self.exit_code != 1:
                raise ValueError("startup_failed requires exit_code == 1")
            if not self.error_id or not self.error_id.strip():
                raise ValueError("startup_failed requires a non-empty error_id")
            return self
        if self.exit_kind == "checkpoint_failed":
            if self.exit_code != 1:
                raise ValueError("checkpoint_failed requires exit_code == 1")
            if self.error_id != "checkpoint_write_failed":
                raise ValueError(
                    "checkpoint_failed requires error_id == checkpoint_write_failed"
                )
            if self.final_checkpoint_saved:
                raise ValueError(
                    "checkpoint_failed requires final_checkpoint_saved is False"
                )
            return self
        if self.exit_kind == "runtime_failed":
            if self.exit_code != 1:
                raise ValueError("runtime_failed requires exit_code == 1")
            if not self.error_id or not self.error_id.strip():
                raise ValueError("runtime_failed requires a non-empty error_id")
            return self
        raise ValueError(f"unsupported exit_kind: {self.exit_kind}")


def _is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _resolve_file_watcher_watch_roots(
    settings: LocalWorkspaceBackendSettings,
) -> frozenset[str]:
    """Watch only explicit INTERGRAX read roots, not auto staging allowlist dirs."""
    auto_staging_roots = {
        settings.managed_upload_staging_dir,
        settings.web_url_staging_dir,
    }
    explicit_roots = frozenset(settings.allowed_read_roots) - auto_staging_roots
    if explicit_roots:
        return explicit_roots
    return frozenset(settings.allowed_read_roots)


def _canonical_watch_roots(roots: frozenset[str]) -> frozenset[str]:
    if not roots:
        raise FileWatcherSidecarConfigurationError("file_watcher_roots_not_configured")
    canonical: set[str] = set()
    for root in roots:
        stripped = str(root).strip()
        if not stripped:
            raise FileWatcherSidecarConfigurationError(
                "file_watcher_roots_not_configured"
            )
        resolved = Path(stripped).expanduser().resolve(strict=False)
        canonical.add(str(resolved))
    if not canonical:
        raise FileWatcherSidecarConfigurationError("file_watcher_roots_not_configured")
    return frozenset(canonical)


def _resolve_absolute_data_home(
    data_home: str,
    *,
    working_directory: Path | None,
) -> Path:
    stripped = data_home.strip()
    if not stripped:
        raise FileWatcherSidecarConfigurationError("file_watcher_data_home_invalid")
    candidate = Path(stripped).expanduser()
    base = working_directory if working_directory is not None else Path.cwd()
    try:
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (base / candidate).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise FileWatcherSidecarConfigurationError(
            "file_watcher_data_home_invalid"
        ) from None
    if not resolved.is_absolute():
        raise FileWatcherSidecarConfigurationError("file_watcher_data_home_invalid")
    return resolved


def build_file_watcher_sidecar_config(
    settings: LocalWorkspaceBackendSettings,
    *,
    working_directory: Path | None = None,
) -> FileWatcherSidecarConfig:
    """Build validated sidecar config from application settings."""
    if not settings.file_watcher_enabled:
        raise FileWatcherSidecarConfigurationError("file_watcher_disabled")

    tenant_id = settings.file_watcher_tenant_id.strip()
    workspace_id = settings.file_watcher_workspace_id.strip()
    collection_id = settings.file_watcher_collection_id.strip()
    if not tenant_id or not workspace_id or not collection_id:
        raise FileWatcherSidecarConfigurationError(
            "file_watcher_identity_not_configured"
        )

    roots = _canonical_watch_roots(_resolve_file_watcher_watch_roots(settings))

    poll_interval = float(settings.file_watcher_poll_interval_seconds)
    if not _is_finite_positive(poll_interval):
        raise FileWatcherSidecarConfigurationError("file_watcher_poll_interval_invalid")

    debounce = float(settings.file_watcher_debounce_seconds)
    if not _is_finite_positive(debounce):
        raise FileWatcherSidecarConfigurationError("file_watcher_debounce_invalid")

    max_wait = float(settings.file_watcher_max_batch_wait_seconds)
    if not _is_finite_positive(max_wait) or max_wait < debounce:
        raise FileWatcherSidecarConfigurationError(
            "file_watcher_max_batch_wait_invalid"
        )

    priority = settings.file_watcher_priority.strip()
    if priority not in _VALID_PRIORITIES:
        raise FileWatcherSidecarConfigurationError("file_watcher_priority_invalid")

    absolute_data_home = _resolve_absolute_data_home(
        settings.data_home,
        working_directory=working_directory,
    )
    try:
        checkpoint_path = file_watcher_checkpoint_path(absolute_data_home)
    except RuntimeError:
        raise FileWatcherSidecarConfigurationError(
            "file_watcher_data_home_invalid"
        ) from None

    runtime_config = FileWatcherRuntimeConfig(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        collection_id=collection_id,
        allowed_roots=roots,
        debounce_seconds=debounce,
        max_batch_wait_seconds=max_wait,
        priority=priority,
    )
    return FileWatcherSidecarConfig(
        runtime_config=runtime_config,
        poll_interval_seconds=poll_interval,
        checkpoint_path=checkpoint_path,
    )


def _is_recognized_snapshot_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    return str(exc) in _SNAPSHOT_ERROR_IDS


def _as_checkpoint_store(
    store: FileWatcherCheckpointStore,
) -> JsonFileWatcherCheckpointStore:
    return cast(JsonFileWatcherCheckpointStore, store)


class FileWatcherSidecar:
    """Foreground process loop for the LKW file watcher."""

    def __init__(
        self,
        *,
        config: FileWatcherSidecarConfig,
        runtime: FileWatcherRuntime,
        checkpoint_store: FileWatcherCheckpointStore,
        control: HostedApplicationControlCoordinator,
        signal_bridge: HostedApplicationSignalBridge,
        monotonic_clock: MonotonicClock,
        sleeper: FileWatcherSleeper,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._checkpoint_store = checkpoint_store
        self._control = control
        self._signal_bridge = signal_bridge
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._logger = logger

    def run(self) -> FileWatcherSidecarResult:
        signals_installed = False
        result: FileWatcherSidecarResult | None = None
        restore_failed = False
        try:
            try:
                self._signal_bridge.install()
            except Exception:
                return FileWatcherSidecarResult(
                    exit_kind="startup_failed",
                    exit_code=1,
                    error_id="signal_install_failed",
                )
            signals_installed = True
            result = self._run_after_signals()
        finally:
            if signals_installed:
                try:
                    self._signal_bridge.restore()
                except Exception:
                    restore_failed = True
        if restore_failed:
            return FileWatcherSidecarResult(
                exit_kind="runtime_failed",
                exit_code=1,
                error_id="signal_restore_failed",
                restored_from_checkpoint=(
                    result.restored_from_checkpoint if result is not None else False
                ),
                cycles_completed=result.cycles_completed if result is not None else 0,
                last_cycle_status=(
                    result.last_cycle_status if result is not None else None
                ),
                final_checkpoint_saved=(
                    result.final_checkpoint_saved if result is not None else False
                ),
            )
        assert result is not None
        return result

    def _run_after_signals(self) -> FileWatcherSidecarResult:
        restored_from_checkpoint = False
        cycles_completed = 0
        last_cycle_status: FileWatcherCycleStatus | None = None

        now = self._monotonic_clock.monotonic()
        try:
            restored = restore_file_watcher_runtime(
                runtime=self._runtime,
                store=_as_checkpoint_store(self._checkpoint_store),
                now_monotonic=now,
            )
        except Exception:
            return FileWatcherSidecarResult(
                exit_kind="startup_failed",
                exit_code=1,
                error_id="checkpoint_restore_failed",
            )

        if restored:
            restored_from_checkpoint = True
        else:
            try:
                self._runtime.initialize()
            except Exception:
                return FileWatcherSidecarResult(
                    exit_kind="startup_failed",
                    exit_code=1,
                    error_id="initial_snapshot_failed",
                )
            try:
                self._checkpoint_store.save(self._runtime.export_checkpoint())
            except Exception:
                return FileWatcherSidecarResult(
                    exit_kind="checkpoint_failed",
                    exit_code=1,
                    error_id="checkpoint_write_failed",
                    restored_from_checkpoint=False,
                    cycles_completed=0,
                    final_checkpoint_saved=False,
                )
            self._logger.info(
                "file_watcher_baseline_initialized",
                extra={"restored": False},
            )

        while not self._control.is_shutdown_requested():
            now = self._monotonic_clock.monotonic()
            try:
                cycle = self._runtime.poll_once(now_monotonic=now)
            except Exception as exc:
                if _is_recognized_snapshot_error(exc):
                    error_id = str(exc)
                    self._logger.error(
                        "file_watcher_snapshot_failed",
                        extra={"error_id": error_id},
                    )
                    if self._control.is_shutdown_requested():
                        break
                    try:
                        self._sleeper.sleep(self._config.poll_interval_seconds)
                    except Exception:
                        return FileWatcherSidecarResult(
                            exit_kind="runtime_failed",
                            exit_code=1,
                            error_id="file_watcher_sleep_failed",
                            restored_from_checkpoint=restored_from_checkpoint,
                            cycles_completed=cycles_completed,
                            last_cycle_status=last_cycle_status,
                            final_checkpoint_saved=False,
                        )
                    continue
                return FileWatcherSidecarResult(
                    exit_kind="runtime_failed",
                    exit_code=1,
                    error_id="file_watcher_runtime_failed",
                    restored_from_checkpoint=restored_from_checkpoint,
                    cycles_completed=cycles_completed,
                    last_cycle_status=last_cycle_status,
                    final_checkpoint_saved=False,
                )

            cycles_completed += 1
            last_cycle_status = cycle.status
            if cycle.status == "enqueue_failed":
                self._logger.warning(
                    "file_watcher_enqueue_failed",
                    extra={"error_id": cycle.error_id},
                )
            elif cycle.status == "enqueued":
                assert cycle.change_token is not None
                assert cycle.task_id is not None
                assert cycle.provider is not None
                assert cycle.tenant_id is not None
                assert cycle.broker_run_id is not None
                assert cycle.idempotency_key is not None
                record = FileWatcherIngestEnqueuedRecord(
                    change_token=cycle.change_token,
                    task_id=cycle.task_id,
                    provider=cycle.provider,
                    tenant_id=cycle.tenant_id,
                    broker_run_id=cycle.broker_run_id,
                    idempotency_key=cycle.idempotency_key,
                )
                print(record.model_dump_json(), flush=True)
                self._logger.info(
                    "file_watcher_ingest_enqueued",
                    extra={
                        "cycle_status": cycle.status,
                        "detected_count": cycle.detected_change_count,
                        "actionable_count": cycle.actionable_path_count,
                        "deleted_count": cycle.deleted_path_count,
                        "cycles_completed": cycles_completed,
                        "restored": restored_from_checkpoint,
                    },
                )
            else:
                self._logger.info(
                    "file_watcher_cycle_completed",
                    extra={
                        "cycle_status": cycle.status,
                        "detected_count": cycle.detected_change_count,
                        "pending_count": cycle.pending_change_count,
                        "actionable_count": cycle.actionable_path_count,
                        "deleted_count": cycle.deleted_path_count,
                        "cycles_completed": cycles_completed,
                        "restored": restored_from_checkpoint,
                    },
                )

            try:
                self._checkpoint_store.save(self._runtime.export_checkpoint())
            except Exception:
                return FileWatcherSidecarResult(
                    exit_kind="checkpoint_failed",
                    exit_code=1,
                    error_id="checkpoint_write_failed",
                    restored_from_checkpoint=restored_from_checkpoint,
                    cycles_completed=cycles_completed,
                    last_cycle_status=last_cycle_status,
                    final_checkpoint_saved=False,
                )

            if self._control.is_shutdown_requested():
                break
            try:
                self._sleeper.sleep(self._config.poll_interval_seconds)
            except Exception:
                return FileWatcherSidecarResult(
                    exit_kind="runtime_failed",
                    exit_code=1,
                    error_id="file_watcher_sleep_failed",
                    restored_from_checkpoint=restored_from_checkpoint,
                    cycles_completed=cycles_completed,
                    last_cycle_status=last_cycle_status,
                    final_checkpoint_saved=False,
                )

        try:
            self._checkpoint_store.save(self._runtime.export_checkpoint())
        except Exception:
            return FileWatcherSidecarResult(
                exit_kind="checkpoint_failed",
                exit_code=1,
                error_id="checkpoint_write_failed",
                restored_from_checkpoint=restored_from_checkpoint,
                cycles_completed=cycles_completed,
                last_cycle_status=last_cycle_status,
                final_checkpoint_saved=False,
            )

        return FileWatcherSidecarResult(
            exit_kind="clean_stop",
            exit_code=0,
            restored_from_checkpoint=restored_from_checkpoint,
            cycles_completed=cycles_completed,
            last_cycle_status=last_cycle_status,
            final_checkpoint_saved=True,
            error_id=None,
        )


def build_local_workspace_file_watcher_sidecar(
    *,
    settings: LocalWorkspaceBackendSettings,
    message_bus: MessageBus,
    monotonic_clock: MonotonicClock | None = None,
    sleeper: FileWatcherSleeper | None = None,
    signal_bridge: HostedApplicationSignalBridge | None = None,
    control: HostedApplicationControlCoordinator | None = None,
    logger: logging.Logger | None = None,
    working_directory: Path | None = None,
) -> FileWatcherSidecar:
    """Compose production sidecar dependencies from settings and message bus."""
    config = build_file_watcher_sidecar_config(
        settings,
        working_directory=working_directory,
    )
    wiring_context = ToolWiringContext(message_bus=message_bus)
    runtime = build_file_watcher_runtime(
        config=config.runtime_config,
        wiring_context=wiring_context,
    )
    checkpoint_store: FileWatcherCheckpointStore = JsonFileWatcherCheckpointStore(
        config.checkpoint_path
    )
    resolved_control = control or HostedApplicationControlCoordinator(
        clock=_UtcWallClock()
    )
    resolved_bridge = signal_bridge or PortableForegroundSignalAdapter(
        coordinator=resolved_control,
        enable_sighup_restart=False,
    )
    return FileWatcherSidecar(
        config=config,
        runtime=runtime,
        checkpoint_store=checkpoint_store,
        control=resolved_control,
        signal_bridge=resolved_bridge,
        monotonic_clock=monotonic_clock or SystemMonotonicClock(),
        sleeper=sleeper or SystemFileWatcherSleeper(),
        logger=logger or logging.getLogger("local_workspace_application.file_watcher"),
    )


def run_local_workspace_file_watcher_sidecar(
    *,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> FileWatcherSidecarResult:
    """Composition root: settings → Kafka bus → sidecar → structured result."""
    resolved: LocalWorkspaceBackendSettings
    if settings is None:
        resolved = cast(
            LocalWorkspaceBackendSettings,
            LocalWorkspaceBackendSettings.from_env(),
        )
    else:
        resolved = settings
    if not resolved.file_watcher_enabled:
        return FileWatcherSidecarResult(
            exit_kind="disabled",
            exit_code=2,
            error_id="file_watcher_disabled",
            cycles_completed=0,
        )
    if not local_workspace_message_bus_enabled():
        return FileWatcherSidecarResult(
            exit_kind="configuration_error",
            exit_code=2,
            error_id="message_bus_not_enabled",
            cycles_completed=0,
        )
    try:
        message_bus = create_local_workspace_kafka_message_bus()
    except Exception:
        return FileWatcherSidecarResult(
            exit_kind="startup_failed",
            exit_code=1,
            error_id="message_bus_initialization_failed",
        )
    try:
        sidecar = build_local_workspace_file_watcher_sidecar(
            settings=resolved,
            message_bus=message_bus,
        )
    except FileWatcherSidecarConfigurationError as exc:
        return FileWatcherSidecarResult(
            exit_kind="configuration_error",
            exit_code=2,
            error_id=str(exc),
            cycles_completed=0,
        )
    return sidecar.run()
