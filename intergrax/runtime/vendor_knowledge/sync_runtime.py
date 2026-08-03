# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compose durable Vendor Knowledge sync onto DocumentStoreTaskQueue/Worker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryResult,
    DocumentRecord,
    DocumentStore,
)
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store import (
    DocumentStoreTaskQueue,
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBindingService
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeFacade
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeSyncSink,
    KnowledgeSyncSinkReceiptInspector,
)
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeReconciliationCandidateInventoryRepository,
    DocumentStoreKnowledgeReconciliationRunRepository,
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

_QUEUE_NAMESPACE_TEMPLATE = "vendor_knowledge.sync_queue.v1:{tenant_id}:"


class _NamespacedDocumentStore:
    """Map logical DocumentStore partitions onto a deterministic physical prefix."""

    def __init__(self, document_store: DocumentStore, *, namespace: str) -> None:
        self._store = document_store
        self._namespace = namespace

    def _physical(self, logical_partition: str) -> str:
        return f"{self._namespace}{logical_partition}"

    def _to_logical(self, document: DocumentRecord) -> DocumentRecord:
        partition = document.partition_key
        if partition.startswith(self._namespace):
            logical = partition[len(self._namespace) :]
        else:
            logical = partition
        return DocumentRecord(
            partition_key=logical,
            row_key=document.row_key,
            data=document.data,
            ttl_seconds=document.ttl_seconds,
        )

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        record = self._store.get(self._physical(partition_key), row_key)
        if record is None:
            return None
        return self._to_logical(record)

    def put(self, document: DocumentRecord) -> None:
        self._store.put(
            DocumentRecord(
                partition_key=self._physical(document.partition_key),
                row_key=document.row_key,
                data=document.data,
                ttl_seconds=document.ttl_seconds,
            )
        )

    def delete(self, partition_key: str, row_key: str) -> None:
        self._store.delete(self._physical(partition_key), row_key)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        result = self._store.query(
            self._physical(partition_key),
            limit=limit,
            row_key_prefix=row_key_prefix,
        )
        documents = tuple(self._to_logical(doc) for doc in result.documents)
        return DocumentQueryResult(documents=documents, total=result.total)

    def close(self) -> None:
        return None


class _TenantVendorKnowledgeTaskQueue(DocumentStoreTaskQueue):
    """DocumentStoreTaskQueue bound to one tenant and Vendor Knowledge task type."""

    def __init__(self, *, document_store: DocumentStore, tenant_id: str) -> None:
        cleaned = tenant_id.strip()
        if not cleaned:
            raise ValueError("tenant_id must be a non-empty string")
        self._bound_tenant_id = cleaned
        namespaced = _NamespacedDocumentStore(
            document_store,
            namespace=_QUEUE_NAMESPACE_TEMPLATE.format(tenant_id=cleaned),
        )
        super().__init__(namespaced)

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        if request.tenant_id != self._bound_tenant_id:
            raise ValueError("vendor knowledge task queue tenant mismatch")
        if request.task_name != VENDOR_KNOWLEDGE_SYNC_TASK_NAME:
            raise ValueError("vendor knowledge task queue task type mismatch")
        return super().enqueue(request)


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
    reconciliation_run_repository: DocumentStoreKnowledgeReconciliationRunRepository
    candidate_inventory_repository: (
        DocumentStoreKnowledgeReconciliationCandidateInventoryRepository
    )
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
    checkpoint_repository = DocumentStoreKnowledgeSyncCheckpointRepository(
        document_store
    )
    item_state_repository = DocumentStoreKnowledgeRemoteItemStateRepository(
        document_store
    )
    reconciliation_run_repository = DocumentStoreKnowledgeReconciliationRunRepository(
        document_store
    )
    candidate_inventory_repository = (
        DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(document_store)
    )
    sink_receipt_inspector = (
        sink if isinstance(sink, KnowledgeSyncSinkReceiptInspector) else None
    )

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
        reconciliation_run_repository=reconciliation_run_repository,
        candidate_inventory_repository=candidate_inventory_repository,
        sink_receipt_inspector=sink_receipt_inspector,
    )

    task_queue = _TenantVendorKnowledgeTaskQueue(
        document_store=document_store,
        tenant_id=tenant_id,
    )
    scheduler = VendorKnowledgeSyncScheduler(task_queue=task_queue)
    registry = TaskExecutionRegistry()

    runtime_holder: dict[str, VendorKnowledgeSyncRuntime] = {}

    def _main_loop_provider() -> asyncio.AbstractEventLoop | None:
        runtime = runtime_holder.get("runtime")
        if runtime is None:
            return None
        return runtime.main_loop()

    def _coordinator_resolver(
        resolved_tenant_id: str,
        run_id: str,
    ) -> VendorKnowledgeSyncCoordinator:
        _ = run_id
        if resolved_tenant_id != tenant_id:
            raise ValueError("vendor knowledge sync tenant mismatch")
        return coordinator

    register_vendor_knowledge_sync_worker_handler(
        registry,
        coordinator_resolver=_coordinator_resolver,
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
        reconciliation_run_repository=reconciliation_run_repository,
        candidate_inventory_repository=candidate_inventory_repository,
    )
    runtime_holder["runtime"] = runtime
    return runtime
