# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compose durable Vendor Knowledge sync onto DocumentStoreTaskQueue/Worker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBindingService
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeFacade
from intergrax.runtime.vendor_knowledge.sync_contracts import KnowledgeSyncSink
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
)
from intergrax.runtime.vendor_knowledge.sync_worker import (
    register_vendor_knowledge_sync_worker_handler,
)


@dataclass(slots=True)
class VendorKnowledgeSyncRuntime:
    task_queue: DocumentStoreTaskQueue
    worker: DocumentStoreTaskWorker
    registry: TaskExecutionRegistry
    scheduler: VendorKnowledgeSyncScheduler
    coordinator: VendorKnowledgeSyncCoordinator
    lease_repository: DocumentStoreKnowledgeSourceLeaseRepository
    checkpoint_repository: DocumentStoreKnowledgeSyncCheckpointRepository
    item_state_repository: DocumentStoreKnowledgeRemoteItemStateRepository
    _main_loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)

    def bind_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

    def main_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._main_loop

    def start(self) -> None:
        self.worker.start()
        self._started = True

    def stop(self) -> None:
        self.worker.stop()
        self._started = False


def build_vendor_knowledge_sync_runtime(
    *,
    document_store: DocumentStore,
    binding_service: KnowledgeSourceBindingService,
    facade: VendorKnowledgeFacade,
    sink: KnowledgeSyncSink,
    owner_id: str,
    tenant_id: str,
    lease_ttl_seconds: int = 60,
    poll_interval_seconds: float = 0.25,
    claim_limit: int = 4,
    retry_delays_seconds: tuple[float, ...] = (0.25, 1.0, 4.0),
) -> VendorKnowledgeSyncRuntime:
    """Wire DocumentStore repositories, coordinator, queue, worker and recovery."""
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError(
            "vendor knowledge sync repositories require ConditionalDocumentStore"
        )

    lease_repository = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repository = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    item_state_repository = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)

    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id=tenant_id,
        owner_id=owner_id,
        binding_service=binding_service,
        facade=facade,
        lease_repository=lease_repository,
        checkpoint_repository=checkpoint_repository,
        item_state_repository=item_state_repository,
        sink=sink,
        lease_ttl_seconds=lease_ttl_seconds,
    )

    task_queue = DocumentStoreTaskQueue(document_store)
    scheduler = VendorKnowledgeSyncScheduler(task_queue=task_queue)
    registry = TaskExecutionRegistry()

    runtime_holder: dict[str, VendorKnowledgeSyncRuntime] = {}

    def _main_loop_provider() -> asyncio.AbstractEventLoop | None:
        runtime = runtime_holder.get("runtime")
        if runtime is None:
            return None
        return runtime.main_loop()

    register_vendor_knowledge_sync_worker_handler(
        registry,
        coordinator=coordinator,
        scheduler=scheduler,
        main_loop_provider=_main_loop_provider,
        retry_delays_seconds=retry_delays_seconds,
    )

    def _on_interrupted(_handle: TaskHandle, request: TaskRequest) -> None:
        if request.task_name != VENDOR_KNOWLEDGE_SYNC_TASK_NAME:
            return
        try:
            job = decode_vendor_knowledge_sync_job(request.payload)
            if job.tenant_id != request.tenant_id:
                return
            scheduler.enqueue_recovery(interrupted_job=job, run_id=request.run_id)
        except Exception:
            return

    worker = DocumentStoreTaskWorker(
        task_queue,
        registry,
        poll_interval_seconds=poll_interval_seconds,
        claim_limit=claim_limit,
        on_interrupted=_on_interrupted,
    )
    runtime = VendorKnowledgeSyncRuntime(
        task_queue=task_queue,
        worker=worker,
        registry=registry,
        scheduler=scheduler,
        coordinator=coordinator,
        lease_repository=lease_repository,
        checkpoint_repository=checkpoint_repository,
        item_state_repository=item_state_repository,
    )
    runtime_holder["runtime"] = runtime
    return runtime
