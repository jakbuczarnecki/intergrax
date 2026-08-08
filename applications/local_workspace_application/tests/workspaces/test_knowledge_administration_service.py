# © Artur Czarnecki. All rights reserved.

"""Tests for provider-neutral natural-language knowledge administration."""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from local_workspace_application.workspaces.knowledge_administration_service import (
    DeterministicKnowledgeAdministrationIntentInterpreter,
    HmacKnowledgeAdministrationConfirmationCodec,
    KnowledgeAccessModeV1,
    KnowledgeAdministrationActionV1,
    KnowledgeAdministrationConfirmationV1,
    KnowledgeAdministrationFilterV1,
    KnowledgeAdministrationIntentV1,
    KnowledgeAdministrationService,
    KnowledgeAdministrationStatusV1,
    Sha256KnowledgeAdministrationIdempotencyKeyFactory,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeInventoryItemV1,
    KnowledgeInventorySummaryV1,
    KnowledgeInventoryV1,
    KnowledgeOperationError,
    KnowledgeOperationResultV1,
    KnowledgeOperationV1,
    KnowledgeRevisionKindV1,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_OTHER_WORKSPACE = "workspace-b"
_INDEXED_ID = "indexed:confluence"
_LIVE_ID = "live:drive"


def _item(
    item_id: str = _INDEXED_ID,
    *,
    label: str = "Confluence",
    mode: KnowledgeAccessModeV1 = KnowledgeAccessModeV1.INDEXED,
    state: str = "active",
    revision: int = 3,
) -> KnowledgeInventoryItemV1:
    actions = (
        KnowledgeOperationV1.SYNC,
        KnowledgeOperationV1.DISABLE,
        KnowledgeOperationV1.DETACH,
    )
    if state == "disabled":
        actions = (KnowledgeOperationV1.ENABLE, KnowledgeOperationV1.DETACH)
    if mode is KnowledgeAccessModeV1.LIVE:
        actions = (KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH)
    return KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=item_id,
        mode=mode,
        display_label=label,
        lifecycle_state=state,
        enabled=state != "disabled",
        detached=False,
        runtime_available=True if mode is KnowledgeAccessModeV1.LIVE else None,
        revision=revision,
        revision_kind=(
            KnowledgeRevisionKindV1.LIFECYCLE
            if mode is KnowledgeAccessModeV1.INDEXED
            else KnowledgeRevisionKindV1.CONFIGURATION
        ),
        available_actions=actions,
        updated_at=_NOW,
    )


def _inventory(
    items: tuple[KnowledgeInventoryItemV1, ...],
    *,
    workspace_id: str = _WORKSPACE,
) -> KnowledgeInventoryV1:
    return KnowledgeInventoryV1(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        items=items,
        summary=KnowledgeInventorySummaryV1(
            total=len(items),
            indexed=sum(item.mode is KnowledgeAccessModeV1.INDEXED for item in items),
            live=sum(item.mode is KnowledgeAccessModeV1.LIVE for item in items),
            active=sum(item.lifecycle_state == "active" for item in items),
            disabled=sum(item.lifecycle_state == "disabled" for item in items),
            attention_required=sum(item.lifecycle_state == "error" for item in items),
        ),
        updated_at=_NOW,
    )


class _Interpreter:
    def __init__(self, intent: KnowledgeAdministrationIntentV1) -> None:
        self.intent = intent
        self.context = None

    async def interpret(self, *, utterance: str, context):
        del utterance
        self.context = context
        return self.intent


class _Inspection:
    def __init__(self, inventory: KnowledgeInventoryV1) -> None:
        self.inventory = inventory

    def list_items(self, *, tenant_id: str, workspace_id: str) -> KnowledgeInventoryV1:
        if tenant_id != self.inventory.tenant_id or workspace_id != self.inventory.workspace_id:
            return _inventory((), workspace_id=workspace_id)
        return self.inventory

    def get_item(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        knowledge_item_id: str,
    ) -> KnowledgeInventoryItemV1:
        if tenant_id != self.inventory.tenant_id or workspace_id != self.inventory.workspace_id:
            raise RuntimeError("not found")
        for item in self.inventory.items:
            if item.knowledge_item_id == knowledge_item_id:
                return item
        raise RuntimeError("not found")


class _Operations:
    def __init__(self, inspection: _Inspection) -> None:
        self.inspection = inspection
        self.commands = []
        self.conflict = False

    async def execute(self, command):
        self.commands.append(command)
        if self.conflict:
            raise KnowledgeOperationError("knowledge_operation_conflict")
        item = self.inspection.get_item(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            knowledge_item_id=command.knowledge_item_id,
        )
        return KnowledgeOperationResultV1(
            item=item,
            operation=command.operation,
            operation_id="operation-1",
            mutation_id="mutation-1",
        )


def _service(
    inventory: KnowledgeInventoryV1,
    intent: KnowledgeAdministrationIntentV1,
    *,
    operations: _Operations | None = None,
    confirmation_port=None,
) -> tuple[KnowledgeAdministrationService, _Inspection, _Operations, _Interpreter]:
    inspection = _Inspection(inventory)
    operation_service = operations or _Operations(inspection)
    interpreter = _Interpreter(intent)
    service = KnowledgeAdministrationService(
        inspection_service=inspection,
        operations_service=operation_service,
        interpreter=interpreter,
        idempotency_key_factory=Sha256KnowledgeAdministrationIdempotencyKeyFactory(),
        confirmation_port=confirmation_port
        or HmacKnowledgeAdministrationConfirmationCodec(secret=b"test-secret"),
    )
    return service, inspection, operation_service, interpreter


@pytest.mark.asyncio
async def test_list_context_is_bounded_and_filters_current_inventory() -> None:
    inventory = _inventory(
        (
            _item(),
            _item(
                _LIVE_ID,
                label="Drive",
                mode=KnowledgeAccessModeV1.LIVE,
                state="disabled",
            ),
        )
    )
    intent = KnowledgeAdministrationIntentV1(
        action=KnowledgeAdministrationActionV1.LIST,
        requested_filter=KnowledgeAdministrationFilterV1.DISABLED,
    )
    service, _, _, interpreter = _service(inventory, intent)

    result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-1",
        utterance="which sources are disabled?",
    )

    assert result.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert [item.knowledge_item_id for item in result.inventory.items] == [_LIVE_ID]
    assert interpreter.context.tenant_id == _TENANT
    assert interpreter.context.items[0].knowledge_item_id == _INDEXED_ID
    assert not hasattr(interpreter.context.items[0], "connection_ref")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "status", "message_code"),
    [
        (
            KnowledgeAdministrationIntentV1(
                action=KnowledgeAdministrationActionV1.SHOW,
                target_text="Confluence",
            ),
            KnowledgeAdministrationStatusV1.AMBIGUOUS,
            "knowledge_admin_target_ambiguous",
        ),
        (
            KnowledgeAdministrationIntentV1(
                action=KnowledgeAdministrationActionV1.SHOW,
                target_text="missing",
            ),
            KnowledgeAdministrationStatusV1.NOT_FOUND,
            "knowledge_admin_target_not_found",
        ),
        (
            KnowledgeAdministrationIntentV1(
                action=KnowledgeAdministrationActionV1.SHOW,
                requested_item_id="indexed:does-not-exist",
            ),
            KnowledgeAdministrationStatusV1.NOT_FOUND,
            "knowledge_admin_target_not_found",
        ),
    ],
)
async def test_target_resolution_is_deterministic(
    intent: KnowledgeAdministrationIntentV1,
    status: KnowledgeAdministrationStatusV1,
    message_code: str,
) -> None:
    inventory = _inventory(
        (
            _item(),
            _item(
                _LIVE_ID,
                label="Confluence",
                mode=KnowledgeAccessModeV1.LIVE,
            ),
        )
    )
    service, _, _, _ = _service(inventory, intent)

    result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-2",
        utterance="ignored",
    )

    assert result.status is status
    assert result.message_code == message_code
    if status is KnowledgeAdministrationStatusV1.AMBIGUOUS:
        assert {item.mode for item in result.candidates} == {
            KnowledgeAccessModeV1.INDEXED,
            KnowledgeAccessModeV1.LIVE,
        }


@pytest.mark.asyncio
async def test_mode_qualification_and_operations_boundary() -> None:
    inventory = _inventory(
        (
            _item(),
            _item(
                _LIVE_ID,
                label="Confluence",
                mode=KnowledgeAccessModeV1.LIVE,
            ),
        )
    )
    intent = KnowledgeAdministrationIntentV1(
        action=KnowledgeAdministrationActionV1.SYNC,
        target_text="Confluence",
        requested_mode=KnowledgeAccessModeV1.INDEXED,
    )
    service, _, operations, _ = _service(inventory, intent)

    result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-3",
        utterance="sync indexed Confluence",
    )

    assert result.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert operations.commands[0].knowledge_item_id == _INDEXED_ID
    assert operations.commands[0].expected_revision == 3
    assert len(operations.commands[0].idempotency_key_hash) == 64


@pytest.mark.asyncio
async def test_unsupported_live_sync_and_concurrent_conflict_are_rejected() -> None:
    intent = KnowledgeAdministrationIntentV1(
        action=KnowledgeAdministrationActionV1.SYNC,
        target_text="Drive",
        requested_mode=KnowledgeAccessModeV1.LIVE,
    )
    service, _, operations, _ = _service(
        _inventory(
            (
                _item(
                    _LIVE_ID,
                    label="Drive",
                    mode=KnowledgeAccessModeV1.LIVE,
                ),
            )
        ),
        intent,
    )
    result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-4",
        utterance="sync live Drive",
    )
    assert result.message_code == "knowledge_admin_action_not_available"
    assert operations.commands == []

    conflict_intent = intent.model_copy(
        update={
            "action": KnowledgeAdministrationActionV1.DISABLE,
            "target_text": "Confluence",
            "requested_mode": KnowledgeAccessModeV1.INDEXED,
        }
    )
    conflict_service, _, conflict_operations, _ = _service(
        _inventory((_item(),)),
        conflict_intent,
    )
    conflict_operations.conflict = True
    conflict_result = await conflict_service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-5",
        utterance="disable Confluence",
    )
    assert conflict_result.status is KnowledgeAdministrationStatusV1.CONFLICT
    assert conflict_result.message_code == "knowledge_admin_conflict"


@pytest.mark.asyncio
async def test_detach_confirmation_is_signed_bound_stale_and_restart_safe() -> None:
    codec = HmacKnowledgeAdministrationConfirmationCodec(secret=b"test-secret")
    intent = KnowledgeAdministrationIntentV1(
        action=KnowledgeAdministrationActionV1.DETACH,
        target_text="Confluence",
    )
    service, inspection, operations, _ = _service(
        _inventory((_item(),)),
        intent,
        confirmation_port=codec,
    )
    pending = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-6",
        utterance="detach Confluence",
    )
    assert pending.status is KnowledgeAdministrationStatusV1.CONFIRMATION_REQUIRED
    assert pending.confirmation_token
    assert operations.commands == []

    confirmed = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-7",
        utterance="detach Confluence",
        confirmation_token=pending.confirmation_token,
    )
    assert confirmed.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert operations.commands[0].operation is KnowledgeOperationV1.DETACH

    tampered = pending.confirmation_token[:-1] + (
        "A" if pending.confirmation_token[-1] != "A" else "B"
    )
    tampered_result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-8",
        utterance="detach Confluence",
        confirmation_token=tampered,
    )
    assert tampered_result.message_code == "knowledge_admin_confirmation_invalid"

    expired = codec.issue(
        KnowledgeAdministrationConfirmationV1(
            token="",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_item_id=_INDEXED_ID,
            operation=KnowledgeOperationV1.DETACH,
            expected_revision=3,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    expired_result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-9",
        utterance="detach Confluence",
        confirmation_token=expired,
    )
    assert expired_result.message_code == "knowledge_admin_confirmation_expired"

    inspection.inventory = _inventory((_item(revision=4),))
    stale_result = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-10",
        utterance="detach Confluence",
        confirmation_token=pending.confirmation_token,
    )
    assert stale_result.status is KnowledgeAdministrationStatusV1.CONFLICT
    assert stale_result.message_code == "knowledge_admin_confirmation_stale"

    restarted_service, _, restarted_operations, _ = _service(
        _inventory((_item(),)),
        intent,
        confirmation_port=codec,
    )
    restarted_result = await restarted_service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-11",
        utterance="detach Confluence",
        confirmation_token=pending.confirmation_token,
    )
    assert restarted_result.status is KnowledgeAdministrationStatusV1.COMPLETED
    assert restarted_operations.commands[0].knowledge_item_id == _INDEXED_ID


@pytest.mark.asyncio
async def test_confirmation_cannot_cross_workspace_and_parser_is_replaceable() -> None:
    codec = HmacKnowledgeAdministrationConfirmationCodec(secret=b"test-secret")
    detach_intent = KnowledgeAdministrationIntentV1(
        action=KnowledgeAdministrationActionV1.DETACH,
        target_text="Confluence",
    )
    service, _, _, _ = _service(_inventory((_item(),)), detach_intent, confirmation_port=codec)
    pending = await service.handle(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        request_id="request-12",
        utterance="detach Confluence",
    )

    other_item = _item()
    other_item = other_item.model_copy(update={"workspace_id": _OTHER_WORKSPACE})
    other_inventory = _inventory((other_item,), workspace_id=_OTHER_WORKSPACE)
    other_service, _, other_operations, _ = _service(
        other_inventory,
        detach_intent,
        confirmation_port=codec,
    )
    result = await other_service.handle(
        tenant_id=_TENANT,
        workspace_id=_OTHER_WORKSPACE,
        request_id="request-13",
        utterance="detach Confluence",
        confirmation_token=pending.confirmation_token,
    )
    assert result.message_code == "knowledge_admin_confirmation_invalid"
    assert other_operations.commands == []

    deterministic = DeterministicKnowledgeAdministrationIntentInterpreter()
    parsed = await deterministic.interpret(
        utterance="sync indexed Confluence source",
        context=service._context(
            inventory=_inventory((_item(),)),
            request_id="request-14",
        ),
    )
    assert parsed.action is KnowledgeAdministrationActionV1.SYNC
    assert parsed.requested_mode is KnowledgeAccessModeV1.INDEXED


def test_administration_module_imports_only_the_public_inspection_operations_facade() -> None:
    path = Path(__file__).parents[2] / "workspaces" / "knowledge_administration_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules <= {
        "__future__",
        "local_workspace_application.workspaces.knowledge_inspection_operations_service",
        "pydantic",
        "typing",
        "enum",
        "datetime",
        "json",
        "hashlib",
        "hmac",
        "base64",
        "binascii",
    }


def test_confirmation_codec_rejects_empty_secret_and_invalid_contract() -> None:
    with pytest.raises(ValueError):
        HmacKnowledgeAdministrationConfirmationCodec(secret=b"")
    with pytest.raises(ValidationError):
        KnowledgeAdministrationConfirmationV1(
            token="",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_item_id=_INDEXED_ID,
            operation=KnowledgeOperationV1.DETACH,
            expected_revision=-1,
            expires_at=_NOW,
        )
