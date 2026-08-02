# © Artur Czarnecki. All rights reserved.

"""Persistence helpers for connected source delivery receipts."""

from __future__ import annotations

from datetime import UTC, datetime

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliveryStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


def _utc_now() -> datetime:
    return datetime.now(UTC)


def delivery_receipt_already_applied(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    delivery_id: str,
) -> bool:
    existing = repository.get_connected_source_delivery_receipt(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        delivery_id=delivery_id,
    )
    return existing is not None


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
) -> ConnectedSourceDeliveryReceipt | None:
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
        status=ConnectedSourceDeliveryStatus.COMPLETED,
        created_at=now,
        completed_at=None,
    )
    if not repository.put_connected_source_delivery_receipt_if_absent(receipt):
        return repository.get_connected_source_delivery_receipt(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            delivery_id=delivery_id,
        )
    return receipt


def complete_delivery_receipt(
    *,
    repository: ManagedWorkspaceRepository,
    receipt: ConnectedSourceDeliveryReceipt,
    documents_indexed: int,
    documents_unchanged: int,
    items_failed: int,
) -> ConnectedSourceDeliveryReceipt:
    completed = receipt.model_copy(
        update={
            "documents_indexed": documents_indexed,
            "documents_unchanged": documents_unchanged,
            "items_failed": items_failed,
            "completed_at": _utc_now(),
        }
    )
    repository.put_connected_source_delivery_receipt(completed)
    return completed
