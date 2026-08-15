# © Artur Czarnecki. All rights reserved.

"""Derived first-run setup snapshot from durable workspace product state (LKW-PRODUCT-3C)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from local_workspace_application.host.readiness import LocalWorkspaceReadinessProvider
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInventoryError,
    KnowledgeInventoryItemV1,
    KnowledgeInventoryV1,
    KnowledgeInspectionService,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from pydantic import BaseModel, ConfigDict, Field


class SetupPhaseV1(StrEnum):
    NO_KNOWLEDGE = "no_knowledge"
    CONFIGURING = "configuring"
    SYNCING = "syncing"
    ATTENTION_REQUIRED = "attention_required"
    READY = "ready"


class SetupNextActionV1(StrEnum):
    ADD_SOURCE = "add_source"
    WAIT_FOR_SYNC = "wait_for_sync"
    RETRY_OR_FIX_SOURCE = "retry_or_fix_source"
    ASK_QUESTION = "ask_question"
    NONE = "none"


_ACTIVE_OPERATION_STATUSES = frozenset(
    {
        WorkspaceOperationStatus.ACCEPTED,
        WorkspaceOperationStatus.QUEUED,
        WorkspaceOperationStatus.RUNNING,
        WorkspaceOperationStatus.PROCESSING,
    }
)


class SetupKnowledgeSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(ge=0)
    indexed: int = Field(ge=0)
    live: int = Field(ge=0)
    active: int = Field(ge=0)
    disabled: int = Field(ge=0)
    attention_required: int = Field(ge=0)
    usable: int = Field(ge=0)


class SetupAttentionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_item_id: str
    error_code: str | None = None
    available_actions: tuple[str, ...] = ()


class SetupRecentOperationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    operation_type: str
    status: str
    error_code: str | None = None


class WorkspaceSetupSnapshotV1(BaseModel):
    """Read-only, derived first-run orchestration contract; safe to recompute."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    workspace_exists: bool = True
    host_ready: bool
    phase: SetupPhaseV1
    can_ask: bool
    has_usable_knowledge: bool
    sync_in_progress: bool
    attention_required: bool
    knowledge_summary: SetupKnowledgeSummaryV1
    recent_operation: SetupRecentOperationV1 | None = None
    attention: SetupAttentionV1 | None = None
    next_action: SetupNextActionV1
    suggested_question: str | None = None
    updated_at: datetime


def _item_needs_attention(item: KnowledgeInventoryItemV1) -> bool:
    return item.lifecycle_state in {"error", "detach_blocked"} or (
        item.runtime_available is False and item.enabled and not item.detached
    )


def _item_syncing(item: KnowledgeInventoryItemV1) -> bool:
    if item.lifecycle_state == "syncing":
        return True
    return item.sync_state in {"queued", "running"}


def _indexed_usable(item: KnowledgeInventoryItemV1) -> bool:
    if item.mode is not KnowledgeAccessModeV1.INDEXED or not item.enabled or item.detached:
        return False
    if item.lifecycle_state not in {"ready", "active"}:
        return False
    return item.sync_state == "succeeded"


def _live_usable(item: KnowledgeInventoryItemV1) -> bool:
    if item.mode is not KnowledgeAccessModeV1.LIVE or not item.enabled or item.detached:
        return False
    if item.lifecycle_state not in {"ready", "active"}:
        return False
    return item.runtime_available is True


def _item_usable(item: KnowledgeInventoryItemV1) -> bool:
    if item.mode is KnowledgeAccessModeV1.INDEXED:
        return _indexed_usable(item)
    return _live_usable(item)


def _operation_active(operation: WorkspaceOperation) -> bool:
    return operation.status in _ACTIVE_OPERATION_STATUSES


def _suggested_question(items: tuple[KnowledgeInventoryItemV1, ...]) -> str | None:
    labels = sorted(
        {
            item.display_label.strip()
            for item in items
            if item.display_label and item.display_label.strip()
        }
    )
    if labels:
        return f"What information is available in {labels[0]}?"
    return "What are the key points in my connected knowledge?"


def _next_action_for_phase(
    phase: SetupPhaseV1,
    *,
    can_ask: bool,
) -> SetupNextActionV1:
    if phase is SetupPhaseV1.NO_KNOWLEDGE:
        return SetupNextActionV1.ADD_SOURCE
    if phase is SetupPhaseV1.SYNCING:
        return SetupNextActionV1.WAIT_FOR_SYNC
    if phase is SetupPhaseV1.ATTENTION_REQUIRED:
        return SetupNextActionV1.RETRY_OR_FIX_SOURCE
    if phase is SetupPhaseV1.READY:
        return SetupNextActionV1.ASK_QUESTION if can_ask else SetupNextActionV1.NONE
    if phase is SetupPhaseV1.CONFIGURING:
        return SetupNextActionV1.WAIT_FOR_SYNC
    return SetupNextActionV1.NONE


def _recent_operation(
    operations: tuple[WorkspaceOperation, ...],
) -> SetupRecentOperationV1 | None:
    if not operations:
        return None
    operation = operations[0]
    return SetupRecentOperationV1(
        operation_id=operation.operation_id,
        operation_type=operation.operation_type.value,
        status=operation.status.value,
        error_code=operation.error_code,
    )


def _attention_for_items(
    items: tuple[KnowledgeInventoryItemV1, ...],
) -> SetupAttentionV1 | None:
    attention_items = sorted(
        (item for item in items if _item_needs_attention(item)),
        key=lambda item: item.knowledge_item_id,
    )
    if not attention_items:
        return None
    item = attention_items[0]
    return SetupAttentionV1(
        knowledge_item_id=item.knowledge_item_id,
        error_code=item.last_error_code,
        available_actions=tuple(action.value for action in item.available_actions),
    )


class WorkspaceSetupSnapshotService:
    """Derives setup snapshot from inventory, operations, and host readiness."""

    def __init__(
        self,
        *,
        workspace_service: ManagedWorkspaceService,
        inspection_service: KnowledgeInspectionService,
        readiness_provider: LocalWorkspaceReadinessProvider,
    ) -> None:
        self._workspace_service = workspace_service
        self._inspection = inspection_service
        self._readiness = readiness_provider

    def derive_snapshot(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceSetupSnapshotV1:
        inventory = self._inspection.list_items(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        operations = tuple(
            self._workspace_service.list_workspace_operations(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                limit=20,
            )
        )
        return self._derive_from_state(
            workspace_id=workspace_id,
            inventory=inventory,
            operations=operations,
        )

    def _derive_from_state(
        self,
        *,
        workspace_id: str,
        inventory: KnowledgeInventoryV1,
        operations: tuple[WorkspaceOperation, ...],
    ) -> WorkspaceSetupSnapshotV1:
        readiness = self._readiness.readiness_snapshot()
        host_ready = readiness.accepts_new_work
        items = inventory.items

        usable_count = sum(_item_usable(item) for item in items)
        has_usable_knowledge = usable_count > 0
        attention_items = tuple(item for item in items if _item_needs_attention(item))
        attention_required = bool(attention_items)
        inventory_syncing = any(_item_syncing(item) for item in items)
        operation_syncing = any(_operation_active(operation) for operation in operations)
        sync_in_progress = inventory_syncing or operation_syncing

        if attention_required:
            phase = SetupPhaseV1.ATTENTION_REQUIRED
        elif sync_in_progress:
            phase = SetupPhaseV1.SYNCING
        elif not items:
            phase = SetupPhaseV1.NO_KNOWLEDGE
        elif has_usable_knowledge:
            phase = SetupPhaseV1.READY
        else:
            phase = SetupPhaseV1.CONFIGURING

        can_ask = host_ready and has_usable_knowledge
        next_action = _next_action_for_phase(phase, can_ask=can_ask)
        suggested_question = (
            _suggested_question(items) if phase is SetupPhaseV1.READY else None
        )

        return WorkspaceSetupSnapshotV1(
            workspace_id=workspace_id,
            workspace_exists=True,
            host_ready=host_ready,
            phase=phase,
            can_ask=can_ask,
            has_usable_knowledge=has_usable_knowledge,
            sync_in_progress=sync_in_progress,
            attention_required=attention_required,
            knowledge_summary=SetupKnowledgeSummaryV1(
                total=inventory.summary.total,
                indexed=inventory.summary.indexed,
                live=inventory.summary.live,
                active=inventory.summary.active,
                disabled=inventory.summary.disabled,
                attention_required=inventory.summary.attention_required,
                usable=usable_count,
            ),
            recent_operation=_recent_operation(operations),
            attention=_attention_for_items(items),
            next_action=next_action,
            suggested_question=suggested_question,
            updated_at=inventory.updated_at,
        )
