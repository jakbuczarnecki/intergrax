# © Artur Czarnecki. All rights reserved.

"""Persistence helpers for connected source delivery receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliveryStatus,
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


@dataclass(frozen=True, slots=True)
class ConnectedSourceDeliveryApplyResult:
    documents_indexed: int
    documents_unchanged: int
    items_processed: int
    items_failed: int
    replayed: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _receipt_core_identity_matches(
    receipt: ConnectedSourceDeliveryReceipt,
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
    delivery_id: str,
    binding_configuration_version: int,
) -> bool:
    return (
        receipt.tenant_id == tenant_id
        and receipt.workspace_id == workspace_id
        and receipt.source_id == source_id
        and receipt.indexed_source_binding_id == indexed_source_binding_id
        and receipt.knowledge_source_binding_ref == knowledge_source_binding_ref
        and receipt.delivery_id == delivery_id
        and receipt.binding_configuration_version == binding_configuration_version
    )


def _receipt_identity_conflict(
    receipt: ConnectedSourceDeliveryReceipt,
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
    delivery_id: str,
    binding_configuration_version: int,
) -> bool:
    return not _receipt_core_identity_matches(
        receipt,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        delivery_id=delivery_id,
        binding_configuration_version=binding_configuration_version,
    )


def delivery_receipt_completed(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    delivery_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
    binding_configuration_version: int,
    operation_id: str,
) -> ConnectedSourceDeliveryReceipt | None:
    existing = repository.get_connected_source_delivery_receipt(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        delivery_id=delivery_id,
    )
    if existing is None:
        return None
    if _receipt_identity_conflict(
        existing,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        delivery_id=delivery_id,
        binding_configuration_version=binding_configuration_version,
    ):
        raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")
    if existing.status is ConnectedSourceDeliveryStatus.COMPLETED:
        if existing.completed_at is None or existing.items_failed != 0:
            raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_invalid")
        _ = operation_id
        return existing
    if existing.status is ConnectedSourceDeliveryStatus.IN_PROGRESS:
        return None
    raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")


def begin_delivery_receipt(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
    delivery_id: str,
    binding_configuration_version: int,
    operation_id: str,
) -> ConnectedSourceDeliveryReceipt:
    existing = repository.get_connected_source_delivery_receipt(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        delivery_id=delivery_id,
    )
    if existing is not None:
        if _receipt_identity_conflict(
            existing,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
            delivery_id=delivery_id,
            binding_configuration_version=binding_configuration_version,
        ):
            raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")
        if existing.status is ConnectedSourceDeliveryStatus.COMPLETED:
            _ = operation_id
            return existing
        if existing.status is ConnectedSourceDeliveryStatus.IN_PROGRESS:
            _ = operation_id
            return existing
        raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")

    now = _utc_now()
    receipt = ConnectedSourceDeliveryReceipt(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        delivery_id=delivery_id,
        binding_configuration_version=binding_configuration_version,
        operation_id=operation_id,
        status=ConnectedSourceDeliveryStatus.IN_PROGRESS,
        created_at=now,
        completed_at=None,
    )
    if not repository.put_connected_source_delivery_receipt_if_absent(receipt):
        reloaded = repository.get_connected_source_delivery_receipt(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            delivery_id=delivery_id,
        )
        if reloaded is None:
            raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")
        return begin_delivery_receipt(
            repository=repository,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
            delivery_id=delivery_id,
            binding_configuration_version=binding_configuration_version,
            operation_id=operation_id,
        )
    return receipt


def complete_delivery_receipt(
    *,
    repository: ManagedWorkspaceRepository,
    receipt: ConnectedSourceDeliveryReceipt,
    documents_indexed: int,
    documents_unchanged: int,
    items_processed: int,
    items_failed: int,
) -> ConnectedSourceDeliveryReceipt:
    _ = items_processed
    if receipt.status is ConnectedSourceDeliveryStatus.COMPLETED:
        return receipt
    if receipt.status is not ConnectedSourceDeliveryStatus.IN_PROGRESS:
        raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")
    if items_failed != 0:
        raise ConnectedSourceSyncSinkError("connected_source_delivery_items_failed")

    completed = receipt.model_copy(
        update={
            "status": ConnectedSourceDeliveryStatus.COMPLETED,
            "documents_indexed": documents_indexed,
            "documents_unchanged": documents_unchanged,
            "items_failed": items_failed,
            "completed_at": _utc_now(),
        }
    )
    if not repository.complete_connected_source_delivery_receipt_if_in_progress(
        expected=receipt,
        replacement=completed,
    ):
        reloaded = repository.get_connected_source_delivery_receipt(
            tenant_id=receipt.tenant_id,
            workspace_id=receipt.workspace_id,
            source_id=receipt.source_id,
            delivery_id=receipt.delivery_id,
        )
        if (
            reloaded is not None
            and reloaded.status is ConnectedSourceDeliveryStatus.COMPLETED
            and _receipt_core_identity_matches(
                reloaded,
                tenant_id=receipt.tenant_id,
                workspace_id=receipt.workspace_id,
                source_id=receipt.source_id,
                indexed_source_binding_id=receipt.indexed_source_binding_id,
                knowledge_source_binding_ref=receipt.knowledge_source_binding_ref,
                delivery_id=receipt.delivery_id,
                binding_configuration_version=receipt.binding_configuration_version,
            )
            and reloaded.completed_at is not None
            and reloaded.items_failed == 0
        ):
            return reloaded
        raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")
    return completed
