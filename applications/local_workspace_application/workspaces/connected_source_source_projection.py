# © Artur Czarnecki. All rights reserved.

"""Repair WorkspaceSource status from persisted connected-source sync operations."""

from __future__ import annotations

from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_ACTIVE_OPERATION_STATUSES = {
    WorkspaceOperationStatus.QUEUED,
    WorkspaceOperationStatus.RUNNING,
}


def _operation_sort_key(operation: WorkspaceOperation) -> tuple[object, ...]:
    return (
        operation.created_at or operation.started_at,
        operation.operation_id,
    )


def _terminal_sort_key(operation: WorkspaceOperation) -> tuple[object, ...]:
    return (
        operation.completed_at,
        operation.created_at or operation.started_at,
        operation.operation_id,
    )


def list_connected_source_sync_operations(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
) -> tuple[WorkspaceOperation, ...]:
    operations = [
        operation
        for operation in repository.list_operations(tenant_id=tenant_id)
        if operation.workspace_id == workspace_id
        and operation.source_id == source_id
        and operation.operation_type is WorkspaceOperationType.SOURCE_SYNC
    ]
    return tuple(sorted(operations, key=_operation_sort_key))


def project_connected_source_source_status(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
) -> WorkspaceSourceStatus | None:
    operations = list_connected_source_sync_operations(
        repository=repository,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
    )
    if any(operation.status in _ACTIVE_OPERATION_STATUSES for operation in operations):
        return WorkspaceSourceStatus.SYNCING

    completed = [
        operation
        for operation in operations
        if operation.status is WorkspaceOperationStatus.COMPLETED and operation.completed_at is not None
    ]
    if completed:
        latest_completed = max(completed, key=_terminal_sort_key)
        _ = latest_completed
        return WorkspaceSourceStatus.READY

    failed = [
        operation
        for operation in operations
        if operation.status is WorkspaceOperationStatus.FAILED and operation.completed_at is not None
    ]
    if failed:
        return WorkspaceSourceStatus.ERROR

    return None


def repair_connected_source_source_projection(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
) -> WorkspaceSource | None:
    source = repository.get_source(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
    )
    if source is None or source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
        return None

    projected_status = project_connected_source_source_status(
        repository=repository,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
    )
    if projected_status is None:
        return source

    operations = list_connected_source_sync_operations(
        repository=repository,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
    )
    last_sync_at = source.last_sync_at
    if projected_status is WorkspaceSourceStatus.READY:
        completed = [
            operation
            for operation in operations
            if operation.status is WorkspaceOperationStatus.COMPLETED and operation.completed_at is not None
        ]
        if completed:
            last_sync_at = max(completed, key=_terminal_sort_key).completed_at

    if source.status is projected_status and (
        projected_status is not WorkspaceSourceStatus.READY or source.last_sync_at == last_sync_at
    ):
        return source

    updated = source.model_copy(
        update={
            "status": projected_status,
            "last_sync_at": last_sync_at,
        }
    )
    repository.put_source(updated)
    return updated


def repair_connected_source_source_projections_for_tenant(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
) -> int:
    repaired = 0
    seen: set[tuple[str, str]] = set()
    for operation in repository.list_operations(tenant_id=tenant_id):
        if operation.operation_type is not WorkspaceOperationType.SOURCE_SYNC:
            continue
        key = (operation.workspace_id, operation.source_id)
        if key in seen:
            continue
        seen.add(key)
        before = repository.get_source(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        after = repair_connected_source_source_projection(
            repository=repository,
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        if after is not None and before is not None and (
            before.status != after.status or before.last_sync_at != after.last_sync_at
        ):
            repaired += 1
    return repaired
