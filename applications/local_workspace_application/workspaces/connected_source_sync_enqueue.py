# © Artur Czarnecki. All rights reserved.

"""Durable enqueue intent for connected-source sync operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.tools.registry.wiring import ToolWiringContext
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
from local_workspace_application.workspaces.sync_enqueue import enqueue_managed_workspace_sync
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob

logger = logging.getLogger(__name__)


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
    existing = repository.get_connected_source_sync_enqueue_intent(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
    )
    generation = 1 if existing is None else existing.enqueue_generation + 1
    intent = ConnectedSourceSyncEnqueueIntent(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        operation_id=operation.operation_id,
        enqueue_generation=generation,
        last_enqueued_generation=existing.last_enqueued_generation if existing else 0,
        updated_at=_utc_now(),
    )
    repository.put_connected_source_sync_enqueue_intent(intent)
    return intent


def try_enqueue_connected_source_sync(
    *,
    repository: ManagedWorkspaceRepository,
    wiring_context: ToolWiringContext,
    operation: WorkspaceOperation,
    intent: ConnectedSourceSyncEnqueueIntent | None = None,
) -> ConnectedSourceEnqueueResult:
    resolved_intent = intent or repository.get_connected_source_sync_enqueue_intent(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
    )
    if resolved_intent is None:
        resolved_intent = record_connected_source_enqueue_intent(
            repository=repository,
            operation=operation,
        )
    if resolved_intent.last_enqueued_generation >= resolved_intent.enqueue_generation:
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
        enqueue_managed_workspace_sync(
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
    ):
        reloaded = repository.get_connected_source_sync_enqueue_intent(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
        )
        generation = reloaded.enqueue_generation if reloaded is not None else resolved_intent.enqueue_generation
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
        from local_workspace_application.workspaces.models import WorkspaceSource

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
                if intent.last_enqueued_generation < intent.enqueue_generation:
                    result = try_enqueue_connected_source_sync(
                        repository=repository,
                        wiring_context=wiring_context,
                        operation=working,
                        intent=intent,
                    )
                    if result.enqueued or result.error is None:
                        operations_repaired += 1
                elif working.status is WorkspaceOperationStatus.QUEUED:
                    result = try_enqueue_connected_source_sync(
                        repository=repository,
                        wiring_context=wiring_context,
                        operation=working,
                        intent=intent,
                    )
                    if result.enqueued:
                        operations_repaired += 1
            except Exception:  # noqa: BLE001 - fail-closed per operation
                errors += 1

    return operations_seen, operations_repaired, errors
