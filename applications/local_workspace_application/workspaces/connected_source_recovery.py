# © Artur Czarnecki. All rights reserved.

"""Startup recovery for interrupted connected-source synchronization operations."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.models import (
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_enqueue import enqueue_managed_workspace_sync
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob


@dataclass(frozen=True, slots=True)
class ConnectedSourceRecoveryResult:
    operations_seen: int = 0
    operations_requeued: int = 0
    errors: int = 0


class ConnectedSourceRecoveryService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        wiring_context: ToolWiringContext,
        tenant_ids: tuple[str, ...] = (),
    ) -> None:
        self._repository = repository
        self._wiring_context = wiring_context
        self._tenant_ids = tenant_ids

    def recover_running_operations(self) -> ConnectedSourceRecoveryResult:
        operations_seen = 0
        operations_requeued = 0
        errors = 0

        for tenant_id in self._tenant_ids:
            if not tenant_id.strip():
                continue
            for operation in self._repository.list_operations(tenant_id=tenant_id):
                if operation.operation_type is not WorkspaceOperationType.SOURCE_SYNC:
                    continue
                if operation.status is not WorkspaceOperationStatus.RUNNING:
                    continue
                operations_seen += 1
                source = self._repository.get_source(
                    tenant_id=operation.tenant_id,
                    workspace_id=operation.workspace_id,
                    source_id=operation.source_id,
                )
                if source is None or source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
                    continue
                try:
                    requeued = operation.model_copy(
                        update={
                            "status": WorkspaceOperationStatus.QUEUED,
                            "error": None,
                        }
                    )
                    self._repository.put_operation(requeued)
                    enqueue_managed_workspace_sync(
                        self._wiring_context,
                        ManagedWorkspaceSyncJob(
                            tenant_id=operation.tenant_id,
                            workspace_id=operation.workspace_id,
                            source_id=operation.source_id,
                            operation_id=operation.operation_id,
                        ),
                    )
                    if source.status is WorkspaceSourceStatus.ERROR:
                        self._repository.put_source(
                            source.model_copy(update={"status": WorkspaceSourceStatus.SYNCING})
                        )
                    operations_requeued += 1
                except Exception:  # noqa: BLE001 - fail-closed per operation
                    errors += 1

        return ConnectedSourceRecoveryResult(
            operations_seen=operations_seen,
            operations_requeued=operations_requeued,
            errors=errors,
        )
