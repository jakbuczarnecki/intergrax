# © Artur Czarnecki. All rights reserved.

"""Durable enqueue intent for connected-source sync operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_enqueue import (
    enqueue_managed_workspace_sync,
)
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskStatus
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
)
from intergrax.tools.registry.wiring import ToolWiringContext

logger = logging.getLogger(__name__)

_MAX_CAS_RETRIES = 3
_TERMINAL_OPERATION_STATUSES = {
    WorkspaceOperationStatus.COMPLETED,
    WorkspaceOperationStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class ConnectedSourceEnqueueResult:
    enqueued: bool
    enqueue_generation: int
    error: str | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def record_connected_source_enqueue_intent(
    *,
    repository: ManagedWorkspaceRepository,
    operation: WorkspaceOperation,
) -> ConnectedSourceSyncEnqueueIntent:
    ownership = repository.resolve_connected_source_ownership_for_source(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
    )
    return repository.allocate_connected_source_sync_enqueue_generation(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        operation_id=operation.operation_id,
        indexed_source_binding_id=ownership[0] if ownership is not None else None,
        knowledge_source_binding_ref=ownership[1] if ownership is not None else None,
        max_attempts=_MAX_CAS_RETRIES,
    )


def _inspect_document_store_task_status(
    queue: DocumentStoreTaskQueue,
    *,
    tenant_id: str,
    task_id: str,
    queue_provider: str,
) -> TaskStatus | None:
    handle = TaskHandle(task_id=task_id, provider=queue_provider, tenant_id=tenant_id)
    return queue.get_status_if_present(handle)


def _latest_task_is_active(
    *,
    wiring_context: ToolWiringContext,
    intent: ConnectedSourceSyncEnqueueIntent,
) -> bool | None:
    if intent.last_task_id is None or intent.last_queue_provider is None:
        return False
    if intent.last_enqueued_generation != intent.enqueue_generation:
        return False
    bus = wiring_context.message_bus
    if isinstance(bus, DocumentStoreTaskQueue):
        status = _inspect_document_store_task_status(
            bus,
            tenant_id=intent.tenant_id,
            task_id=intent.last_task_id,
            queue_provider=intent.last_queue_provider,
        )
        if status is None:
            return False
        return status in {TaskStatus.PENDING, TaskStatus.RUNNING}
    # Non-DocumentStore backends cannot be inspected here; preserve best-effort reuse.
    return None


def _should_allocate_next_generation(
    *,
    wiring_context: ToolWiringContext,
    intent: ConnectedSourceSyncEnqueueIntent,
) -> bool:
    if intent.last_enqueued_generation < intent.enqueue_generation:
        return False
    active = _latest_task_is_active(wiring_context=wiring_context, intent=intent)
    if active is True:
        return False
    return not (
        active is None and intent.last_enqueued_generation == intent.enqueue_generation
    )


def try_enqueue_connected_source_sync(
    *,
    repository: ManagedWorkspaceRepository,
    wiring_context: ToolWiringContext,
    operation: WorkspaceOperation,
    intent: ConnectedSourceSyncEnqueueIntent | None = None,
) -> ConnectedSourceEnqueueResult:
    if operation.status in _TERMINAL_OPERATION_STATUSES:
        return ConnectedSourceEnqueueResult(
            enqueued=False,
            enqueue_generation=0,
            error="operation_terminal",
        )

    resolved_intent = intent or repository.get_connected_source_sync_enqueue_intent(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
    )
    if resolved_intent is None:
        resolved_intent = record_connected_source_enqueue_intent(
            repository=repository,
            operation=operation,
        )

    if _latest_task_is_active(wiring_context=wiring_context, intent=resolved_intent) is True:
        return ConnectedSourceEnqueueResult(
            enqueued=False,
            enqueue_generation=resolved_intent.enqueue_generation,
        )

    if _should_allocate_next_generation(
        wiring_context=wiring_context,
        intent=resolved_intent,
    ):
        resolved_intent = record_connected_source_enqueue_intent(
            repository=repository,
            operation=operation,
        )

    if (
        resolved_intent.last_enqueued_generation >= resolved_intent.enqueue_generation
        and _latest_task_is_active(
            wiring_context=wiring_context,
            intent=resolved_intent,
        )
        is not False
    ):
        return ConnectedSourceEnqueueResult(
            enqueued=False,
            enqueue_generation=resolved_intent.enqueue_generation,
        )

    job = ManagedWorkspaceSyncJob(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        operation_id=operation.operation_id,
    )
    try:
        enqueue_output = enqueue_managed_workspace_sync(
            wiring_context,
            job,
            enqueue_generation=resolved_intent.enqueue_generation,
        )
    except Exception as exc:  # noqa: BLE001 - observable enqueue failure
        logger.warning(
            "connected_source_sync_enqueue_failed operation_id=%s error=%s",
            operation.operation_id,
            exc.__class__.__name__,
        )
        return ConnectedSourceEnqueueResult(
            enqueued=False,
            enqueue_generation=resolved_intent.enqueue_generation,
            error=exc.__class__.__name__,
        )
    if not repository.mark_connected_source_sync_enqueued(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
        expected_generation=resolved_intent.enqueue_generation,
        task_id=enqueue_output.task_id,
        queue_provider=enqueue_output.provider,
    ):
        reloaded = repository.get_connected_source_sync_enqueue_intent(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
        )
        generation = (
            reloaded.enqueue_generation if reloaded is not None else resolved_intent.enqueue_generation
        )
        return ConnectedSourceEnqueueResult(enqueued=False, enqueue_generation=generation)
    return ConnectedSourceEnqueueResult(
        enqueued=True,
        enqueue_generation=resolved_intent.enqueue_generation,
    )


def durable_requeue_connected_source_operation(
    *,
    repository: ManagedWorkspaceRepository,
    wiring_context: ToolWiringContext | None,
    operation: WorkspaceOperation,
    source_status: WorkspaceSourceStatus = WorkspaceSourceStatus.SYNCING,
    error_code: str | None = None,
) -> tuple[WorkspaceOperation, ConnectedSourceEnqueueResult | None]:
    if operation.status in _TERMINAL_OPERATION_STATUSES:
        return operation, None

    intent = record_connected_source_enqueue_intent(repository=repository, operation=operation)
    requeued = operation.model_copy(
        update={
            "status": WorkspaceOperationStatus.QUEUED,
            "error": error_code,
        }
    )
    repository.put_operation(requeued)
    source = repository.get_source(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
    )
    if source is not None and source.status is not source_status:
        repository.put_source(source.model_copy(update={"status": source_status}))
    if wiring_context is None:
        return requeued, None
    enqueue_result = try_enqueue_connected_source_sync(
        repository=repository,
        wiring_context=wiring_context,
        operation=requeued,
        intent=intent,
    )
    return requeued, enqueue_result


def repair_connected_source_pending_enqueue(
    *,
    repository: ManagedWorkspaceRepository,
    wiring_context: ToolWiringContext,
    tenant_ids: tuple[str, ...],
) -> tuple[int, int, int]:
    """Return (operations_seen, operations_repaired, errors)."""
    operations_seen = 0
    operations_repaired = 0
    errors = 0

    terminal_statuses = {
        WorkspaceOperationStatus.COMPLETED,
        WorkspaceOperationStatus.FAILED,
    }

    for tenant_id in tenant_ids:
        if not tenant_id.strip():
            continue
        for operation in repository.list_operations(tenant_id=tenant_id):
            if operation.operation_type is not WorkspaceOperationType.SOURCE_SYNC:
                continue
            if operation.status in terminal_statuses:
                continue
            source = repository.get_source(
                tenant_id=operation.tenant_id,
                workspace_id=operation.workspace_id,
                source_id=operation.source_id,
            )
            if source is None or source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
                continue
            operations_seen += 1
            try:
                working = operation
                if working.status is WorkspaceOperationStatus.RUNNING:
                    working = working.model_copy(
                        update={
                            "status": WorkspaceOperationStatus.QUEUED,
                            "error": None,
                        }
                    )
                    repository.put_operation(working)
                    if source.status is WorkspaceSourceStatus.ERROR:
                        repository.put_source(
                            source.model_copy(update={"status": WorkspaceSourceStatus.SYNCING})
                        )
                intent = repository.get_connected_source_sync_enqueue_intent(
                    tenant_id=working.tenant_id,
                    operation_id=working.operation_id,
                )
                if intent is None:
                    intent = record_connected_source_enqueue_intent(
                        repository=repository,
                        operation=working,
                    )
                if _latest_task_is_active(wiring_context=wiring_context, intent=intent) is True:
                    continue
                if _should_allocate_next_generation(
                    wiring_context=wiring_context,
                    intent=intent,
                ):
                    intent = record_connected_source_enqueue_intent(
                        repository=repository,
                        operation=working,
                    )
                if intent.last_enqueued_generation < intent.enqueue_generation or (
                    working.status is WorkspaceOperationStatus.QUEUED
                    and _latest_task_is_active(wiring_context=wiring_context, intent=intent) is not True
                ):
                    result = try_enqueue_connected_source_sync(
                        repository=repository,
                        wiring_context=wiring_context,
                        operation=working,
                        intent=intent,
                    )
                    if result.enqueued or result.error is None:
                        operations_repaired += 1
            except Exception:  # noqa: BLE001 - fail-closed per operation
                errors += 1

    return operations_seen, operations_repaired, errors
