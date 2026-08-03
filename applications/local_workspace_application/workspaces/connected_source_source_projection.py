# © Artur Czarnecki. All rights reserved.

"""Repair WorkspaceSource status from persisted connected-source sync operations."""

from __future__ import annotations

from datetime import UTC, datetime

from local_workspace_application.workspaces.connected_source_ids import connected_source_id
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationStatusV1,
)
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
_TERMINAL_OPERATION_STATUSES = {
    WorkspaceOperationStatus.COMPLETED,
    WorkspaceOperationStatus.FAILED,
}
_DATETIME_MIN_UTC = datetime.min.replace(tzinfo=UTC)


def _sortable_datetime(value: datetime | None) -> datetime:
    if value is None:
        return _DATETIME_MIN_UTC
    return value


def _operation_started_at(operation: WorkspaceOperation) -> datetime:
    if operation.created_at is not None:
        return operation.created_at
    if operation.started_at is not None:
        return operation.started_at
    return _DATETIME_MIN_UTC


def _operation_sort_key(operation: WorkspaceOperation) -> tuple[datetime, str]:
    return (
        _operation_started_at(operation),
        operation.operation_id,
    )


def _terminal_sort_key(operation: WorkspaceOperation) -> tuple[datetime, datetime, str]:
    return (
        _sortable_datetime(operation.completed_at),
        _operation_started_at(operation),
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

    terminal = [
        operation
        for operation in operations
        if operation.status in _TERMINAL_OPERATION_STATUSES and operation.completed_at is not None
    ]
    if not terminal:
        return None

    latest_terminal = max(terminal, key=_terminal_sort_key)
    if latest_terminal.status is WorkspaceOperationStatus.COMPLETED:
        return WorkspaceSourceStatus.READY
    return WorkspaceSourceStatus.ERROR


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


class ConnectedSourceOriginValidationError(Exception):
    """Raised when a connected Source durable origin cannot be proven."""


def validate_connected_source_durable_origin(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    binding: WorkspaceIndexedSourceBinding,
    committed_configuration_revision: int,
) -> WorkspaceSource:
    if binding.tenant_id != tenant_id or binding.workspace_id != workspace_id:
        raise ConnectedSourceOriginValidationError("tenant_workspace_mismatch")
    if binding.source_id != source_id:
        raise ConnectedSourceOriginValidationError("source_id_mismatch")
    expected_source_id = connected_source_id(
        tenant_id,
        workspace_id,
        binding.knowledge_source_binding_ref,
    )
    if source_id != expected_source_id:
        raise ConnectedSourceOriginValidationError("deterministic_source_id_mismatch")

    source = repository.get_source(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
    )
    if source is None:
        raise ConnectedSourceOriginValidationError("source_missing")
    if (
        source.tenant_id != tenant_id
        or source.workspace_id != workspace_id
        or source.source_id != source_id
    ):
        raise ConnectedSourceOriginValidationError("source_identity_mismatch")
    if source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
        raise ConnectedSourceOriginValidationError("source_type_mismatch")
    if source.path != "" or source.recursive is not False:
        raise ConnectedSourceOriginValidationError("source_shape_mismatch")
    if source.knowledge_configuration_creation_mutation_id is None:
        raise ConnectedSourceOriginValidationError("source_creation_mutation_missing")
    if source.knowledge_configuration_visibility_revision is None:
        raise ConnectedSourceOriginValidationError("source_visibility_revision_missing")
    if source.knowledge_configuration_visibility_revision > binding.effective_revision:
        raise ConnectedSourceOriginValidationError("source_visibility_after_binding")
    if binding.effective_revision > committed_configuration_revision:
        raise ConnectedSourceOriginValidationError("binding_revision_after_committed")

    creation_mutation_id = source.knowledge_configuration_creation_mutation_id
    creation_mutation = repository.find_knowledge_configuration_mutation_by_id(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        mutation_id=creation_mutation_id,
    )
    if creation_mutation is None:
        raise ConnectedSourceOriginValidationError("creation_mutation_missing")
    if creation_mutation.operation is not WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE:
        raise ConnectedSourceOriginValidationError("creation_mutation_wrong_operation")
    if creation_mutation.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED:
        raise ConnectedSourceOriginValidationError("creation_mutation_not_committed")
    if creation_mutation.outcome is not WorkspaceKnowledgeMutationOutcomeV1.APPLIED:
        raise ConnectedSourceOriginValidationError("creation_mutation_wrong_outcome")
    if creation_mutation.target_revision != creation_mutation.committed_revision:
        raise ConnectedSourceOriginValidationError("creation_mutation_revision_mismatch")
    if (
        creation_mutation.committed_revision
        != source.knowledge_configuration_visibility_revision
    ):
        raise ConnectedSourceOriginValidationError("creation_mutation_visibility_mismatch")
    if creation_mutation.result_entity_type != "indexed_source_binding":
        raise ConnectedSourceOriginValidationError("creation_mutation_result_type_mismatch")
    if creation_mutation.result_entity_id != binding.indexed_source_binding_id:
        raise ConnectedSourceOriginValidationError("creation_mutation_result_id_mismatch")
    if creation_mutation.semantic_identity_hash != binding.semantic_identity_hash:
        raise ConnectedSourceOriginValidationError("creation_mutation_semantic_mismatch")

    origin_revision = source.knowledge_configuration_visibility_revision
    origin_binding = repository.get_knowledge_indexed_source_version(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        indexed_source_binding_id=binding.indexed_source_binding_id,
        effective_revision=origin_revision,
    )
    if origin_binding is None:
        raise ConnectedSourceOriginValidationError("origin_binding_missing")
    if origin_binding.mutation_id != creation_mutation_id:
        raise ConnectedSourceOriginValidationError("origin_binding_mutation_mismatch")
    if origin_binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
        raise ConnectedSourceOriginValidationError("origin_binding_not_active")
    if origin_binding.indexed_source_binding_id != binding.indexed_source_binding_id:
        raise ConnectedSourceOriginValidationError("origin_binding_id_mismatch")
    if origin_binding.knowledge_source_binding_ref != binding.knowledge_source_binding_ref:
        raise ConnectedSourceOriginValidationError("origin_binding_ref_mismatch")
    if origin_binding.source_id != binding.source_id:
        raise ConnectedSourceOriginValidationError("origin_binding_source_mismatch")
    if origin_binding.semantic_identity_hash != binding.semantic_identity_hash:
        raise ConnectedSourceOriginValidationError("origin_binding_semantic_mismatch")
    return source
