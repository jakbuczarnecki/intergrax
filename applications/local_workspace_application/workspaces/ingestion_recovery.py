# © Artur Czarnecki. All rights reserved.

"""Startup recovery for active Knowledge Ingestion operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from local_workspace_application.workspaces.knowledge_intake import KnowledgeIntakeService
from local_workspace_application.workspaces.models import (
    WorkspaceOperationStatus,
    WorkspaceSourceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class KnowledgeIngestionRecoveryResult:
    locators_seen: int = 0
    workspaces_reconciled: int = 0
    operations_requeued: int = 0
    processing_failed: int = 0
    stale_locators_removed: int = 0
    errors: int = 0


class KnowledgeIngestionRecoveryService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        knowledge_intake: KnowledgeIntakeService,
    ) -> None:
        self._repository = repository
        self._knowledge_intake = knowledge_intake

    def recover_all(self) -> KnowledgeIngestionRecoveryResult:
        locators_seen = 0
        workspaces_reconciled = 0
        operations_requeued = 0
        processing_failed = 0
        reconcile_groups: dict[tuple[str, str], None] = {}

        scan = self._repository.scan_active_ingestion_locators()
        stale_locators_removed = scan.malformed_removed
        errors = scan.malformed_seen

        for locator in scan.locators:
            locators_seen += 1
            try:
                operation = self._repository.get_operation(
                    tenant_id=locator.tenant_id,
                    operation_id=locator.operation_id,
                )
                if operation is None:
                    self._repository.delete_active_ingestion_locator(locator.operation_id)
                    stale_locators_removed += 1
                    continue

                if operation.status in {
                    WorkspaceOperationStatus.COMPLETED,
                    WorkspaceOperationStatus.FAILED,
                }:
                    self._repository.delete_active_ingestion_locator(locator.operation_id)
                    stale_locators_removed += 1
                    continue

                if operation.status in {
                    WorkspaceOperationStatus.ACCEPTED,
                    WorkspaceOperationStatus.QUEUED,
                }:
                    reconcile_groups[(operation.tenant_id, operation.workspace_id)] = None
                    continue

                if operation.status is WorkspaceOperationStatus.PROCESSING:
                    now = _utc_now()
                    self._repository.put_operation(
                        operation.model_copy(
                            update={
                                "status": WorkspaceOperationStatus.FAILED,
                                "error_code": "interrupted_by_host_restart",
                                "error": "interrupted_by_host_restart",
                                "completed_at": now,
                            }
                        )
                    )
                    source = self._repository.get_source(
                        tenant_id=operation.tenant_id,
                        workspace_id=operation.workspace_id,
                        source_id=operation.source_id,
                    )
                    if source is not None and source.status is WorkspaceSourceStatus.PROCESSING:
                        self._repository.put_source(
                            source.model_copy(update={"status": WorkspaceSourceStatus.ERROR})
                        )
                    self._repository.delete_active_ingestion_locator(locator.operation_id)
                    processing_failed += 1
                    continue

                # Unexpected status — treat locator as stale.
                self._repository.delete_active_ingestion_locator(locator.operation_id)
                stale_locators_removed += 1
            except Exception:  # noqa: BLE001 - fail-closed per locator
                errors += 1

        for tenant_id, workspace_id in reconcile_groups:
            try:
                resumed = self._knowledge_intake.reconcile_workspace(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                )
                workspaces_reconciled += 1
                operations_requeued += resumed
            except Exception:  # noqa: BLE001 - fail-closed per workspace
                errors += 1

        return KnowledgeIngestionRecoveryResult(
            locators_seen=locators_seen,
            workspaces_reconciled=workspaces_reconciled,
            operations_requeued=operations_requeued,
            processing_failed=processing_failed,
            stale_locators_removed=stale_locators_removed,
            errors=errors,
        )
