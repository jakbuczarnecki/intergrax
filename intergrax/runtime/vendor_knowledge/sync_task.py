# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Application-composition adapter for Vendor Knowledge sync."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA,
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    vendor_knowledge_sync_idempotency_key,
)
from intergrax.runtime.vendor_knowledge.sync_worker import (
    MainLoopProvider,
    Sleeper,
    VendorKnowledgeSyncWorkerOutput,
    make_vendor_knowledge_sync_worker_handler,
)

CoordinatorFactory = Callable[[str, str], VendorKnowledgeSyncCoordinator]

__all__ = [
    "VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA",
    "VENDOR_KNOWLEDGE_SYNC_TASK_NAME",
    "VendorKnowledgeSyncDispatcher",
    "VendorKnowledgeSyncJob",
    "VendorKnowledgeSyncWorkerOutput",
    "decode_vendor_knowledge_sync_job",
    "encode_vendor_knowledge_sync_job",
    "make_vendor_knowledge_sync_handler",
    "owner_id_for_sync_run",
    "register_vendor_knowledge_sync_handler",
    "vendor_knowledge_sync_idempotency_key",
]


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def owner_id_for_sync_run(run_id: str) -> str:
    """Stable lease owner identity derived from queue run identity (hashed)."""
    cleaned = _require_non_empty(run_id, field_name="run_id")
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return f"vendor_knowledge.sync:{digest}"


class VendorKnowledgeSyncDispatcher:
    """Thin application wrapper over VendorKnowledgeSyncScheduler."""

    def __init__(self, task_queue: TaskQueue) -> None:
        self._scheduler = VendorKnowledgeSyncScheduler(task_queue=task_queue)

    @property
    def scheduler(self) -> VendorKnowledgeSyncScheduler:
        return self._scheduler

    def enqueue(self, *, job: VendorKnowledgeSyncJob, run_id: str) -> TaskHandle:
        return self._scheduler.enqueue_job(job=job, run_id=run_id)

    def enqueue_incremental(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        operation_id: str,
        run_id: str,
        page_size: int = 100,
    ) -> TaskHandle:
        return self._scheduler.enqueue_incremental(
            tenant_id=tenant_id,
            binding_id=binding_id,
            operation_id=operation_id,
            run_id=run_id,
            page_size=page_size,
        )

    def enqueue_reconciliation(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        operation_id: str,
        run_id: str,
        page_size: int = 100,
    ) -> TaskHandle:
        return self._scheduler.enqueue_reconciliation(
            tenant_id=tenant_id,
            binding_id=binding_id,
            operation_id=operation_id,
            run_id=run_id,
            page_size=page_size,
        )


def make_vendor_knowledge_sync_handler(
    coordinator_factory: CoordinatorFactory,
    dispatcher: VendorKnowledgeSyncDispatcher,
    *,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = (0.25, 1.0, 4.0),
    sleeper: Sleeper | None = None,
):
    """Build a TaskExecutionRegistry-compatible sync handler for applications."""

    def _resolver(tenant_id: str, run_id: str) -> VendorKnowledgeSyncCoordinator:
        return coordinator_factory(tenant_id, owner_id_for_sync_run(run_id))

    return make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=_resolver,
        scheduler=dispatcher.scheduler,
        main_loop_provider=main_loop_provider,
        retry_delays_seconds=retry_delays_seconds,
        sleeper=sleeper,
    )


def register_vendor_knowledge_sync_handler(
    registry: TaskExecutionRegistry,
    coordinator_factory: CoordinatorFactory,
    dispatcher: VendorKnowledgeSyncDispatcher,
    *,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = (0.25, 1.0, 4.0),
    sleeper: Sleeper | None = None,
) -> None:
    """Register the application adapter handler under vendor_knowledge.sync.v1."""
    registry.register(
        VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
        make_vendor_knowledge_sync_handler(
            coordinator_factory,
            dispatcher,
            main_loop_provider=main_loop_provider,
            retry_delays_seconds=retry_delays_seconds,
            sleeper=sleeper,
        ),
    )
