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
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_MAX_CAS_RETRIES = 3
_TERMINAL_OPERATION_STATUSES = {
    WorkspaceOperationStatus.COMPLETED,
    WorkspaceOperationStatus.FAILED,
}


class ConnectedSourceDeliveryAccountingConflictError(RuntimeError):
    def __init__(self, error_code: str = "connected_source_delivery_accounting_conflict") -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class ConnectedSourceDeliveryAccountingResult:
    applied: bool
    documents_indexed: int = 0
    documents_unchanged: int = 0
    items_failed: int = 0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_accounting_matches_receipt(
    accounting: ConnectedSourceOperationDeliveryAccounting,
    receipt_documents_indexed: int,
    receipt_documents_unchanged: int,
    receipt_items_failed: int,
) -> None:
    if (
        accounting.documents_indexed != receipt_documents_indexed
        or accounting.documents_unchanged != receipt_documents_unchanged
        or accounting.items_failed != receipt_items_failed
    ):
        raise ConnectedSourceDeliveryAccountingConflictError()


def _aggregate_accounting_counters(
    accountings: tuple[ConnectedSourceOperationDeliveryAccounting, ...],
) -> tuple[int, int, int]:
    documents_indexed = sum(item.documents_indexed for item in accountings)
    documents_unchanged = sum(item.documents_unchanged for item in accountings)
    items_failed = sum(item.items_failed for item in accountings)
    return documents_indexed, documents_unchanged, items_failed


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
    accounting_created = False
    if existing is None:
        accounting = ConnectedSourceOperationDeliveryAccounting(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
            delivery_id=delivery_id,
            documents_indexed=receipt.documents_indexed,
            documents_unchanged=receipt.documents_unchanged,
            items_failed=receipt.items_failed,
            accounted_at=_utc_now(),
        )
        accounting_created = repository.put_connected_source_delivery_accounting_if_absent(accounting)
        existing = repository.get_connected_source_delivery_accounting(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
            delivery_id=delivery_id,
        )
        if existing is None:
            return operation, ConnectedSourceDeliveryAccountingResult(applied=False)
    _validate_accounting_matches_receipt(
        existing,
        receipt.documents_indexed,
        receipt.documents_unchanged,
        receipt.items_failed,
    )

    accountings = tuple(
        repository.list_connected_source_delivery_accounting(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
        )
    )
    total_indexed, total_unchanged, total_failed = _aggregate_accounting_counters(accountings)

    for _ in range(_MAX_CAS_RETRIES):
        current = repository.get_operation(
            tenant_id=operation.tenant_id,
            operation_id=operation.operation_id,
        )
        if current is None:
            return operation, ConnectedSourceDeliveryAccountingResult(applied=False)

        counters_match = (
            current.documents_indexed == total_indexed
            and current.documents_unchanged == total_unchanged
            and current.files_failed == total_failed
        )
        if counters_match:
            return current, ConnectedSourceDeliveryAccountingResult(
                applied=accounting_created,
                documents_indexed=receipt.documents_indexed,
                documents_unchanged=receipt.documents_unchanged,
                items_failed=receipt.items_failed,
            )

        replacement = current.model_copy(
            update={
                "documents_indexed": total_indexed,
                "documents_unchanged": total_unchanged,
                "files_failed": total_failed,
            }
        )
        if current.status in _TERMINAL_OPERATION_STATUSES:
            replacement = replacement.model_copy(update={"status": current.status})
        elif operation.status in _TERMINAL_OPERATION_STATUSES:
            replacement = replacement.model_copy(update={"status": current.status})

        if repository.replace_operation_if_match(expected=current, replacement=replacement):
            return replacement, ConnectedSourceDeliveryAccountingResult(
                applied=True,
                documents_indexed=receipt.documents_indexed,
                documents_unchanged=receipt.documents_unchanged,
                items_failed=receipt.items_failed,
            )

    reloaded = repository.get_operation(
        tenant_id=operation.tenant_id,
        operation_id=operation.operation_id,
    )
    final_operation = reloaded or operation
    return final_operation, ConnectedSourceDeliveryAccountingResult(applied=False)
