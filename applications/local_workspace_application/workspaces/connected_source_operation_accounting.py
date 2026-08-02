# © Artur Czarnecki. All rights reserved.

"""Durable idempotent operation counters for connected-source deliveries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryApplyResult,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryStatus,
    ConnectedSourceOperationDeliveryAccounting,
)
from local_workspace_application.workspaces.models import WorkspaceOperation
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


@dataclass(frozen=True, slots=True)
class ConnectedSourceDeliveryAccountingResult:
    applied: bool
    documents_indexed: int = 0
    documents_unchanged: int = 0
    items_failed: int = 0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def apply_completed_delivery_accounting(
    *,
    repository: ManagedWorkspaceRepository,
    operation: WorkspaceOperation,
    delivery_id: str,
    sink_result: ConnectedSourceDeliveryApplyResult | None = None,
) -> tuple[WorkspaceOperation, ConnectedSourceDeliveryAccountingResult]:
    receipt = repository.get_connected_source_delivery_receipt(
        tenant_id=operation.tenant_id,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        delivery_id=delivery_id,
    )
    if receipt is None or receipt.status is not ConnectedSourceDeliveryStatus.COMPLETED:
        if sink_result is not None and not sink_result.replayed:
            return operation, ConnectedSourceDeliveryAccountingResult(
                applied=False,
                documents_indexed=sink_result.documents_indexed,
                documents_unchanged=sink_result.documents_unchanged,
                items_failed=sink_result.items_failed,
            )
        return operation, ConnectedSourceDeliveryAccountingResult(applied=False)

    existing = repository.get_connected_source_delivery_accounting(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
        delivery_id=delivery_id,
    )
    if existing is not None:
        return operation, ConnectedSourceDeliveryAccountingResult(applied=False)

    accounting = ConnectedSourceOperationDeliveryAccounting(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
        delivery_id=delivery_id,
        documents_indexed=receipt.documents_indexed,
        documents_unchanged=receipt.documents_unchanged,
        items_failed=receipt.items_failed,
        accounted_at=_utc_now(),
    )
    if not repository.put_connected_source_delivery_accounting_if_absent(accounting):
        return operation, ConnectedSourceDeliveryAccountingResult(applied=False)

    updated = operation.model_copy(
        update={
            "documents_indexed": operation.documents_indexed + receipt.documents_indexed,
            "documents_unchanged": operation.documents_unchanged + receipt.documents_unchanged,
            "files_failed": operation.files_failed + receipt.items_failed,
        }
    )
    repository.put_operation(updated)
    return updated, ConnectedSourceDeliveryAccountingResult(
        applied=True,
        documents_indexed=receipt.documents_indexed,
        documents_unchanged=receipt.documents_unchanged,
        items_failed=receipt.items_failed,
    )
