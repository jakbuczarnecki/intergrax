# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker handler for durable Vendor Knowledge synchronization."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.tools.execution_models import ToolExecutionResult

MainLoopProvider = Callable[[], asyncio.AbstractEventLoop | None]
Sleeper = Callable[[float], None]

_DEFAULT_RETRY_DELAYS: tuple[float, ...] = (0.25, 1.0, 4.0)
_WORKER_OUTPUT_SCHEMA = "vendor_knowledge.sync_worker.v1"


class VendorKnowledgeSyncWorkerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = _WORKER_OUTPUT_SCHEMA
    operation_id: str
    binding_id: str
    mode: KnowledgeSyncMode
    status: KnowledgeSyncRunStatus
    delivery_id: str | None = None
    changes_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    tombstone_count: int = Field(ge=0)
    checkpoint_advanced: bool
    has_more: bool
    continuation_task_id: str | None = None
    execution_attempts: int = Field(ge=1)


def _default_sleeper(delay_seconds: float) -> None:
    import time

    time.sleep(delay_seconds)


def make_vendor_knowledge_sync_worker_handler(
    *,
    coordinator: VendorKnowledgeSyncCoordinator,
    scheduler: VendorKnowledgeSyncScheduler,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
    sleeper: Sleeper | None = None,
):
    sleep = sleeper or _default_sleeper
    delays = tuple(float(item) for item in retry_delays_seconds)

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
    ) -> ToolExecutionResult[VendorKnowledgeSyncWorkerOutput]:
        _ = idempotency_key
        try:
            job = decode_vendor_knowledge_sync_job(payload)
        except (ValueError, ValidationError, TypeError):
            return ToolExecutionResult.fail(
                "vendor_knowledge_sync_invalid_job",
                "vendor knowledge sync job payload is invalid",
            )
        if job.tenant_id != tenant_id:
            return ToolExecutionResult.fail(
                "vendor_knowledge_sync_tenant_mismatch",
                "vendor knowledge sync tenant mismatch",
            )

        attempt = 0
        while True:
            attempt += 1
            try:
                result = _run_coordinator(
                    coordinator=coordinator,
                    job=job,
                    main_loop_provider=main_loop_provider,
                )
            except VendorKnowledgeError as exc:
                if exc.retryable and attempt <= len(delays):
                    sleep(delays[attempt - 1])
                    continue
                if exc.retryable:
                    return ToolExecutionResult.fail(
                        "vendor_knowledge_sync_retry_exhausted",
                        "vendor knowledge sync retry exhausted",
                    )
                return ToolExecutionResult.fail(
                    f"vendor_knowledge_sync_{exc.code.value}",
                    exc.safe_message,
                )
            except Exception:
                if attempt <= len(delays):
                    sleep(delays[attempt - 1])
                    continue
                return ToolExecutionResult.fail(
                    "vendor_knowledge_sync_retry_exhausted",
                    "vendor knowledge sync retry exhausted",
                )

            if result.status is KnowledgeSyncRunStatus.LEASE_BUSY:
                if attempt <= len(delays):
                    sleep(delays[attempt - 1])
                    continue
                return ToolExecutionResult.fail(
                    "vendor_knowledge_sync_retry_exhausted",
                    "vendor knowledge sync retry exhausted",
                )

            continuation_task_id: str | None = None
            if (
                result.status is KnowledgeSyncRunStatus.COMPLETED
                and result.has_more
                and result.delivery_id is not None
            ):
                try:
                    handle = scheduler.enqueue_continuation(
                        parent_job=job,
                        run_id=run_id,
                        trigger_delivery_id=result.delivery_id,
                    )
                    continuation_task_id = handle.task_id
                except Exception:
                    if attempt <= len(delays):
                        sleep(delays[attempt - 1])
                        continue
                    return ToolExecutionResult.fail(
                        "vendor_knowledge_sync_retry_exhausted",
                        "vendor knowledge sync retry exhausted",
                    )

            return ToolExecutionResult.ok(
                VendorKnowledgeSyncWorkerOutput(
                    operation_id=job.operation_id,
                    binding_id=job.binding_id,
                    mode=job.mode,
                    status=result.status,
                    delivery_id=result.delivery_id,
                    changes_count=result.changes_count,
                    active_count=result.active_count,
                    tombstone_count=result.tombstone_count,
                    checkpoint_advanced=result.checkpoint_advanced,
                    has_more=result.has_more,
                    continuation_task_id=continuation_task_id,
                    execution_attempts=attempt,
                )
            )

    return handler


def _run_coordinator(
    *,
    coordinator: VendorKnowledgeSyncCoordinator,
    job: VendorKnowledgeSyncJob,
    main_loop_provider: MainLoopProvider | None,
) -> KnowledgeSyncRunResult:
    if job.mode is KnowledgeSyncMode.INCREMENTAL:
        coro = coordinator.sync_once(
            binding_id=job.binding_id,
            page_size=job.page_size,
        )
    else:
        coro = coordinator.reconcile_once(
            binding_id=job.binding_id,
            page_size=job.page_size,
            restart=job.restart,
        )
    main_loop = main_loop_provider() if main_loop_provider is not None else None
    if main_loop is not None and main_loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, main_loop).result(timeout=600)
    return asyncio.run(coro)


def register_vendor_knowledge_sync_worker_handler(
    registry: TaskExecutionRegistry,
    *,
    coordinator: VendorKnowledgeSyncCoordinator,
    scheduler: VendorKnowledgeSyncScheduler,
    logical_task_name: str = VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
    sleeper: Sleeper | None = None,
) -> None:
    registry.register(
        logical_task_name,
        make_vendor_knowledge_sync_worker_handler(
            coordinator=coordinator,
            scheduler=scheduler,
            main_loop_provider=main_loop_provider,
            retry_delays_seconds=retry_delays_seconds,
            sleeper=sleeper,
        ),
    )
