# © Artur Czarnecki. All rights reserved.

"""Provider-neutral natural-language administration for LKW knowledge items."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import ClassVar, Protocol

from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInspectionService,
    KnowledgeInventoryError,
    KnowledgeInventoryItemV1,
    KnowledgeInventorySummaryV1,
    KnowledgeInventoryV1,
    KnowledgeOperationCommandV1,
    KnowledgeOperationError,
    KnowledgeOperationResultV1,
    KnowledgeOperationsService,
    KnowledgeOperationV1,
)
from pydantic import BaseModel, ConfigDict, Field


class KnowledgeAdministrationActionV1(StrEnum):
    LIST = "list"
    SHOW = "show"
    SYNC = "sync"
    RETRY_SYNC = "retry_sync"
    DISABLE = "disable"
    ENABLE = "enable"
    DETACH = "detach"
    RESUME_DETACH = "resume_detach"


class KnowledgeAdministrationStatusV1(StrEnum):
    COMPLETED = "completed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    REJECTED = "rejected"
    CONFLICT = "conflict"


class KnowledgeAdministrationFilterV1(StrEnum):
    ALL = "all"
    INDEXED = "indexed"
    LIVE = "live"
    ACTIVE = "active"
    DISABLED = "disabled"
    ATTENTION_REQUIRED = "attention_required"


class KnowledgeAdministrationIntentV1(BaseModel):
    """Immutable, provider-neutral interpretation of one administration request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: KnowledgeAdministrationActionV1
    target_text: str | None = None
    requested_mode: KnowledgeAccessModeV1 | None = None
    requested_item_id: str | None = None
    requested_filter: KnowledgeAdministrationFilterV1 | None = None
    retry_operation_id: str | None = None
    confirmation_token: str | None = None


class KnowledgeAdministrationContextItemV1(BaseModel):
    """Safe bounded item context exposed to an interpreter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_item_id: str
    display_label: str | None
    mode: KnowledgeAccessModeV1
    lifecycle_state: str
    enabled: bool
    detached: bool
    available_actions: tuple[KnowledgeOperationV1, ...]


class KnowledgeAdministrationContextV1(BaseModel):
    """Credential-free context for a replaceable intent interpreter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    request_id: str
    items: tuple[KnowledgeAdministrationContextItemV1, ...]


class KnowledgeAdministrationConfirmationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    tenant_id: str
    workspace_id: str
    knowledge_item_id: str
    operation: KnowledgeOperationV1
    expected_revision: int = Field(ge=0)
    expires_at: datetime


class KnowledgeAdministrationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: KnowledgeAdministrationStatusV1
    intent: KnowledgeAdministrationIntentV1
    item: KnowledgeInventoryItemV1 | None = None
    inventory: KnowledgeInventoryV1 | None = None
    candidates: tuple[KnowledgeInventoryItemV1, ...] = ()
    confirmation_required: bool = False
    confirmation_token: str | None = None
    operation_result: KnowledgeOperationResultV1 | None = None
    message_code: str


class KnowledgeAdministrationIntentInterpreterPort(Protocol):
    async def interpret(
        self,
        *,
        utterance: str,
        context: KnowledgeAdministrationContextV1,
    ) -> KnowledgeAdministrationIntentV1:
        """Interpret an utterance without selecting or mutating a target."""
        ...


class KnowledgeAdministrationIdempotencyKeyFactoryPort(Protocol):
    def create(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        knowledge_item_id: str,
        operation: KnowledgeOperationV1,
        request_id: str,
    ) -> str:
        """Return the hash accepted by KnowledgeOperationsService."""
        ...


class KnowledgeAdministrationConfirmationPort(Protocol):
    def issue(self, confirmation: KnowledgeAdministrationConfirmationV1) -> str:
        """Issue a tamper-resistant confirmation token."""
        ...

    def verify(self, token: str) -> KnowledgeAdministrationConfirmationV1:
        """Verify signature, shape and expiry."""
        ...


class KnowledgeAdministrationConfirmationError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class Sha256KnowledgeAdministrationIdempotencyKeyFactory:
    """Stateless request-bound idempotency hash factory."""

    def create(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        knowledge_item_id: str,
        operation: KnowledgeOperationV1,
        request_id: str,
    ) -> str:
        material = (
            f"{tenant_id}\x1f{workspace_id}\x1f{knowledge_item_id}"
            f"\x1f{operation.value}\x1f{request_id}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class HmacKnowledgeAdministrationConfirmationCodec:
    """Signed, stateless confirmation codec with a short expiry."""

    _VERSION = 1

    def __init__(
        self,
        *,
        secret: bytes,
        ttl: timedelta = timedelta(minutes=5),
    ) -> None:
        if not secret:
            raise ValueError("confirmation secret must not be empty")
        if ttl <= timedelta(0):
            raise ValueError("confirmation ttl must be positive")
        self._secret = secret
        self._ttl = ttl

    def issue(self, confirmation: KnowledgeAdministrationConfirmationV1) -> str:
        payload = {
            "v": self._VERSION,
            "tenant_id": confirmation.tenant_id,
            "workspace_id": confirmation.workspace_id,
            "knowledge_item_id": confirmation.knowledge_item_id,
            "operation": confirmation.operation.value,
            "expected_revision": confirmation.expected_revision,
            "expires_at": int(confirmation.expires_at.timestamp()),
        }
        encoded_payload = _urlsafe_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = hmac.new(
            self._secret,
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return f"{encoded_payload}.{_urlsafe_encode(signature)}"

    def verify(self, token: str) -> KnowledgeAdministrationConfirmationV1:
        try:
            encoded_payload, encoded_signature = token.split(".", maxsplit=1)
            expected_signature = hmac.new(
                self._secret,
                encoded_payload.encode("ascii"),
                hashlib.sha256,
            ).digest()
            actual_signature = _urlsafe_decode(encoded_signature)
            if not hmac.compare_digest(actual_signature, expected_signature):
                raise KnowledgeAdministrationConfirmationError(
                    "knowledge_admin_confirmation_invalid"
                )
            payload = json.loads(_urlsafe_decode(encoded_payload))
            if payload.get("v") != self._VERSION:
                raise KnowledgeAdministrationConfirmationError(
                    "knowledge_admin_confirmation_invalid"
                )
            expires_at = datetime.fromtimestamp(int(payload["expires_at"]), tz=UTC)
            if expires_at <= datetime.now(UTC):
                raise KnowledgeAdministrationConfirmationError(
                    "knowledge_admin_confirmation_expired"
                )
            return KnowledgeAdministrationConfirmationV1(
                token=token,
                tenant_id=payload["tenant_id"],
                workspace_id=payload["workspace_id"],
                knowledge_item_id=payload["knowledge_item_id"],
                operation=payload["operation"],
                expected_revision=payload["expected_revision"],
                expires_at=expires_at,
            )
        except KnowledgeAdministrationConfirmationError:
            raise
        except (
            binascii.Error,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise KnowledgeAdministrationConfirmationError(
                "knowledge_admin_confirmation_invalid"
            ) from exc


class DeterministicKnowledgeAdministrationIntentInterpreter:
    """Small explicit-command reference interpreter for tests and local wiring."""

    _ACTION_PREFIXES = (
        ("resume detach", KnowledgeAdministrationActionV1.RESUME_DETACH),
        ("retry sync", KnowledgeAdministrationActionV1.RETRY_SYNC),
        ("sync", KnowledgeAdministrationActionV1.SYNC),
        ("disable", KnowledgeAdministrationActionV1.DISABLE),
        ("enable", KnowledgeAdministrationActionV1.ENABLE),
        ("detach", KnowledgeAdministrationActionV1.DETACH),
    )

    async def interpret(
        self,
        *,
        utterance: str,
        context: KnowledgeAdministrationContextV1,
    ) -> KnowledgeAdministrationIntentV1:
        del context
        text = " ".join(utterance.strip().split())
        lowered = text.casefold()
        if lowered in {"list", "list sources", "show my knowledge sources"}:
            return KnowledgeAdministrationIntentV1(
                action=KnowledgeAdministrationActionV1.LIST,
                requested_filter=KnowledgeAdministrationFilterV1.ALL,
            )
        if lowered.startswith("which sources are "):
            state = lowered.removeprefix("which sources are ").rstrip("?")
            filters = {
                "disabled": KnowledgeAdministrationFilterV1.DISABLED,
                "active": KnowledgeAdministrationFilterV1.ACTIVE,
                "attention-required": KnowledgeAdministrationFilterV1.ATTENTION_REQUIRED,
            }
            if state in filters:
                return KnowledgeAdministrationIntentV1(
                    action=KnowledgeAdministrationActionV1.LIST,
                    requested_filter=filters[state],
                )
        if lowered.startswith("show "):
            target = text[5:].strip()
            if target.casefold().endswith(" source"):
                target = target[:-7].strip()
            return self._targeted_intent(
                KnowledgeAdministrationActionV1.SHOW,
                target,
            )
        for prefix, action in self._ACTION_PREFIXES:
            if lowered == prefix:
                return KnowledgeAdministrationIntentV1(action=action)
            if lowered.startswith(f"{prefix} "):
                return self._targeted_intent(action, text[len(prefix) :].strip())
        raise ValueError("unsupported administration utterance")

    @staticmethod
    def _targeted_intent(
        action: KnowledgeAdministrationActionV1,
        target: str,
    ) -> KnowledgeAdministrationIntentV1:
        if target.casefold().endswith(" source"):
            target = target[:-7].strip()
        requested_mode = None
        for mode_name, mode in (
            ("indexed ", KnowledgeAccessModeV1.INDEXED),
            ("live ", KnowledgeAccessModeV1.LIVE),
        ):
            if target.casefold().startswith(mode_name):
                requested_mode = mode
                target = target[len(mode_name) :].strip()
                break
        return KnowledgeAdministrationIntentV1(
            action=action,
            target_text=target or None,
            requested_mode=requested_mode,
        )


class KnowledgeAdministrationService:
    """Resolves interpreted intent and delegates every mutation to operations."""

    _OPERATION_BY_ACTION: ClassVar[
        dict[KnowledgeAdministrationActionV1, KnowledgeOperationV1]
    ] = {
        KnowledgeAdministrationActionV1.SYNC: KnowledgeOperationV1.SYNC,
        KnowledgeAdministrationActionV1.RETRY_SYNC: KnowledgeOperationV1.RETRY_SYNC,
        KnowledgeAdministrationActionV1.DISABLE: KnowledgeOperationV1.DISABLE,
        KnowledgeAdministrationActionV1.ENABLE: KnowledgeOperationV1.ENABLE,
        KnowledgeAdministrationActionV1.DETACH: KnowledgeOperationV1.DETACH,
        KnowledgeAdministrationActionV1.RESUME_DETACH: KnowledgeOperationV1.RESUME_DETACH,
    }

    def __init__(
        self,
        *,
        inspection_service: KnowledgeInspectionService,
        operations_service: KnowledgeOperationsService,
        interpreter: KnowledgeAdministrationIntentInterpreterPort,
        idempotency_key_factory: KnowledgeAdministrationIdempotencyKeyFactoryPort,
        confirmation_port: KnowledgeAdministrationConfirmationPort,
        confirmation_ttl: timedelta = timedelta(minutes=5),
    ) -> None:
        if confirmation_ttl <= timedelta(0):
            raise ValueError("confirmation ttl must be positive")
        self._inspection = inspection_service
        self._operations = operations_service
        self._interpreter = interpreter
        self._idempotency = idempotency_key_factory
        self._confirmation = confirmation_port
        self._confirmation_ttl = confirmation_ttl

    async def handle(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        request_id: str,
        utterance: str,
        confirmation_token: str | None = None,
    ) -> KnowledgeAdministrationResultV1:
        try:
            inventory = self._inspection.list_items(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except KnowledgeInventoryError as exc:
            return self._error_result(
                self._fallback_intent(),
                status=KnowledgeAdministrationStatusV1.REJECTED,
                message_code=self._map_inventory_error(exc.error_code),
            )
        if (
            inventory.tenant_id != tenant_id
            or inventory.workspace_id != workspace_id
            or any(
                item.tenant_id != tenant_id or item.workspace_id != workspace_id
                for item in inventory.items
            )
        ):
            return self._error_result(
                self._fallback_intent(),
                status=KnowledgeAdministrationStatusV1.NOT_FOUND,
                inventory=inventory,
                message_code="knowledge_admin_target_not_found",
            )

        context = self._context(inventory=inventory, request_id=request_id)
        try:
            intent = await self._interpreter.interpret(
                utterance=utterance,
                context=context,
            )
        except (ValueError, RuntimeError):
            return self._error_result(
                self._fallback_intent(),
                status=KnowledgeAdministrationStatusV1.REJECTED,
                inventory=inventory,
                message_code="knowledge_admin_unavailable",
            )

        supplied_token = confirmation_token or intent.confirmation_token
        if confirmation_token and intent.confirmation_token and not hmac.compare_digest(
            confirmation_token,
            intent.confirmation_token,
        ):
            return self._error_result(
                intent,
                status=KnowledgeAdministrationStatusV1.REJECTED,
                inventory=inventory,
                message_code="knowledge_admin_confirmation_invalid",
            )

        if intent.action is KnowledgeAdministrationActionV1.LIST:
            filtered = self._filter_inventory(inventory, intent)
            return self._result(
                intent,
                status=KnowledgeAdministrationStatusV1.COMPLETED,
                inventory=filtered,
                message_code="knowledge_admin_completed",
            )

        if intent.action is KnowledgeAdministrationActionV1.SHOW:
            resolution = self._resolve_target(inventory, intent)
            if resolution.item is None:
                return self._resolution_result(intent, inventory, resolution)
            return self._result(
                intent,
                status=KnowledgeAdministrationStatusV1.COMPLETED,
                item=resolution.item,
                inventory=inventory,
                message_code="knowledge_admin_completed",
            )

        resolution = self._resolve_target(inventory, intent)
        if resolution.item is None:
            return self._resolution_result(intent, inventory, resolution)
        item = resolution.item
        operation = self._OPERATION_BY_ACTION[intent.action]
        if operation not in item.available_actions:
            return self._action_unavailable(intent, item, inventory)
        if operation is KnowledgeOperationV1.RETRY_SYNC and not intent.retry_operation_id:
            return self._action_unavailable(intent, item, inventory)

        confirmation = None
        if supplied_token is not None:
            confirmation_result = self._verify_confirmation(
                token=supplied_token,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                item=item,
                operation=operation,
                intent=intent,
            )
            if isinstance(confirmation_result, KnowledgeAdministrationResultV1):
                return confirmation_result
            confirmation = confirmation_result
            try:
                item = self._inspection.get_item(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    knowledge_item_id=item.knowledge_item_id,
                )
            except KnowledgeInventoryError as exc:
                return self._error_result(
                    intent,
                    status=KnowledgeAdministrationStatusV1.NOT_FOUND,
                    inventory=inventory,
                    message_code=self._map_inventory_error(exc.error_code),
                )
            if item.revision != confirmation.expected_revision:
                return self._error_result(
                    intent,
                    status=KnowledgeAdministrationStatusV1.CONFLICT,
                    item=item,
                    inventory=inventory,
                    message_code="knowledge_admin_confirmation_stale",
                )
            if operation not in item.available_actions:
                return self._action_unavailable(intent, item, inventory)
        elif operation is KnowledgeOperationV1.DETACH:
            token = self._confirmation.issue(
                KnowledgeAdministrationConfirmationV1(
                    token="",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    knowledge_item_id=item.knowledge_item_id,
                    operation=operation,
                    expected_revision=item.revision,
                    expires_at=datetime.now(UTC) + self._confirmation_ttl,
                )
            )
            return self._result(
                intent,
                status=KnowledgeAdministrationStatusV1.CONFIRMATION_REQUIRED,
                item=item,
                inventory=inventory,
                confirmation_required=True,
                confirmation_token=token,
                message_code="knowledge_admin_confirmation_required",
            )

        command = KnowledgeOperationCommandV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            knowledge_item_id=item.knowledge_item_id,
            operation=operation,
            expected_revision=(
                confirmation.expected_revision if confirmation is not None else item.revision
            ),
            idempotency_key_hash=self._idempotency.create(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                knowledge_item_id=item.knowledge_item_id,
                operation=operation,
                request_id=request_id,
            ),
            operation_id=(
                intent.retry_operation_id
                if operation is KnowledgeOperationV1.RETRY_SYNC
                else None
            ),
        )
        try:
            operation_result = await self._operations.execute(command)
        except KnowledgeOperationError as exc:
            return self._operation_error_result(
                intent,
                inventory=inventory,
                item=item,
                error_code=exc.error_code,
            )
        refreshed_inventory = inventory
        try:
            refreshed_inventory = self._inspection.list_items(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except KnowledgeInventoryError:
            pass
        return self._result(
            intent,
            status=KnowledgeAdministrationStatusV1.COMPLETED,
            item=operation_result.item,
            inventory=refreshed_inventory,
            operation_result=operation_result,
            message_code="knowledge_admin_completed",
        )

    def _verify_confirmation(
        self,
        *,
        token: str,
        tenant_id: str,
        workspace_id: str,
        item: KnowledgeInventoryItemV1,
        operation: KnowledgeOperationV1,
        intent: KnowledgeAdministrationIntentV1,
    ) -> KnowledgeAdministrationConfirmationV1 | KnowledgeAdministrationResultV1:
        try:
            confirmation = self._confirmation.verify(token)
        except KnowledgeAdministrationConfirmationError as exc:
            return self._error_result(
                intent,
                status=KnowledgeAdministrationStatusV1.REJECTED,
                item=item,
                message_code=exc.error_code,
            )
        if (
            confirmation.tenant_id != tenant_id
            or confirmation.workspace_id != workspace_id
            or confirmation.knowledge_item_id != item.knowledge_item_id
            or confirmation.operation is not operation
        ):
            return self._error_result(
                intent,
                status=KnowledgeAdministrationStatusV1.REJECTED,
                item=item,
                message_code="knowledge_admin_confirmation_invalid",
            )
        return confirmation

    def _resolve_target(
        self,
        inventory: KnowledgeInventoryV1,
        intent: KnowledgeAdministrationIntentV1,
    ) -> _TargetResolution:
        items = self._mode_items(inventory.items, intent.requested_mode)
        if intent.requested_item_id is not None:
            matches = tuple(
                item
                for item in items
                if item.knowledge_item_id == intent.requested_item_id
            )
            return _TargetResolution.from_matches(matches)

        target = intent.target_text.strip() if intent.target_text else ""
        if not target:
            return _TargetResolution.not_found()
        stable_matches = tuple(item for item in items if item.knowledge_item_id == target)
        if stable_matches:
            return _TargetResolution.from_matches(stable_matches)
        normalized_target = _normalize(target)
        label_matches = tuple(
            item
            for item in items
            if item.display_label is not None
            and _normalize(item.display_label) == normalized_target
        )
        return _TargetResolution.from_matches(label_matches)

    @staticmethod
    def _mode_items(
        items: tuple[KnowledgeInventoryItemV1, ...],
        mode: KnowledgeAccessModeV1 | None,
    ) -> tuple[KnowledgeInventoryItemV1, ...]:
        if mode is None:
            return items
        return tuple(item for item in items if item.mode is mode)

    @staticmethod
    def _filter_inventory(
        inventory: KnowledgeInventoryV1,
        intent: KnowledgeAdministrationIntentV1,
    ) -> KnowledgeInventoryV1:
        items = KnowledgeAdministrationService._mode_items(
            inventory.items,
            intent.requested_mode,
        )
        requested_filter = intent.requested_filter
        if requested_filter is not None and requested_filter is not KnowledgeAdministrationFilterV1.ALL:
            if requested_filter is KnowledgeAdministrationFilterV1.INDEXED:
                items = tuple(item for item in items if item.mode is KnowledgeAccessModeV1.INDEXED)
            elif requested_filter is KnowledgeAdministrationFilterV1.LIVE:
                items = tuple(item for item in items if item.mode is KnowledgeAccessModeV1.LIVE)
            elif requested_filter is KnowledgeAdministrationFilterV1.ACTIVE:
                items = tuple(item for item in items if item.lifecycle_state == "active")
            elif requested_filter is KnowledgeAdministrationFilterV1.DISABLED:
                items = tuple(item for item in items if item.lifecycle_state == "disabled")
            elif requested_filter is KnowledgeAdministrationFilterV1.ATTENTION_REQUIRED:
                items = tuple(
                    item
                    for item in items
                    if item.lifecycle_state in {"error", "detach_blocked"}
                    or (
                        item.runtime_available is False
                        and item.enabled
                        and not item.detached
                    )
                )
        return _inventory_with_items(inventory, items)

    def _context(
        self,
        *,
        inventory: KnowledgeInventoryV1,
        request_id: str,
    ) -> KnowledgeAdministrationContextV1:
        return KnowledgeAdministrationContextV1(
            tenant_id=inventory.tenant_id,
            workspace_id=inventory.workspace_id,
            request_id=request_id,
            items=tuple(
                KnowledgeAdministrationContextItemV1(
                    knowledge_item_id=item.knowledge_item_id,
                    display_label=item.display_label,
                    mode=item.mode,
                    lifecycle_state=item.lifecycle_state,
                    enabled=item.enabled,
                    detached=item.detached,
                    available_actions=item.available_actions,
                )
                for item in inventory.items
            ),
        )

    @staticmethod
    def _resolution_result(
        intent: KnowledgeAdministrationIntentV1,
        inventory: KnowledgeInventoryV1,
        resolution: _TargetResolution,
    ) -> KnowledgeAdministrationResultV1:
        if resolution.ambiguous:
            return KnowledgeAdministrationResultV1(
                status=KnowledgeAdministrationStatusV1.AMBIGUOUS,
                intent=intent,
                inventory=inventory,
                candidates=resolution.matches,
                message_code="knowledge_admin_target_ambiguous",
            )
        return KnowledgeAdministrationResultV1(
            status=KnowledgeAdministrationStatusV1.NOT_FOUND,
            intent=intent,
            inventory=inventory,
            message_code="knowledge_admin_target_not_found",
        )

    @staticmethod
    def _action_unavailable(
        intent: KnowledgeAdministrationIntentV1,
        item: KnowledgeInventoryItemV1,
        inventory: KnowledgeInventoryV1,
    ) -> KnowledgeAdministrationResultV1:
        return KnowledgeAdministrationResultV1(
            status=KnowledgeAdministrationStatusV1.REJECTED,
            intent=intent,
            item=item,
            inventory=inventory,
            message_code="knowledge_admin_action_not_available",
        )

    @staticmethod
    def _operation_error_result(
        intent: KnowledgeAdministrationIntentV1,
        *,
        inventory: KnowledgeInventoryV1,
        item: KnowledgeInventoryItemV1,
        error_code: str,
    ) -> KnowledgeAdministrationResultV1:
        if error_code == "knowledge_operation_conflict":
            status = KnowledgeAdministrationStatusV1.CONFLICT
            message_code = "knowledge_admin_conflict"
        elif error_code in {
            "knowledge_operation_not_supported",
            "knowledge_operation_invalid_state",
            "knowledge_operation_retry_target_required",
        }:
            status = KnowledgeAdministrationStatusV1.REJECTED
            message_code = "knowledge_admin_action_not_available"
        elif error_code == "knowledge_item_not_found":
            status = KnowledgeAdministrationStatusV1.NOT_FOUND
            message_code = "knowledge_admin_target_not_found"
        else:
            status = KnowledgeAdministrationStatusV1.REJECTED
            message_code = "knowledge_admin_unavailable"
        return KnowledgeAdministrationResultV1(
            status=status,
            intent=intent,
            item=item,
            inventory=inventory,
            message_code=message_code,
        )

    @staticmethod
    def _map_inventory_error(error_code: str) -> str:
        if error_code == "knowledge_item_not_found":
            return "knowledge_admin_target_not_found"
        return "knowledge_admin_unavailable"

    @staticmethod
    def _fallback_intent() -> KnowledgeAdministrationIntentV1:
        return KnowledgeAdministrationIntentV1(
            action=KnowledgeAdministrationActionV1.LIST
        )

    @staticmethod
    def _result(
        intent: KnowledgeAdministrationIntentV1,
        *,
        status: KnowledgeAdministrationStatusV1,
        item: KnowledgeInventoryItemV1 | None = None,
        inventory: KnowledgeInventoryV1 | None = None,
        candidates: tuple[KnowledgeInventoryItemV1, ...] = (),
        confirmation_required: bool = False,
        confirmation_token: str | None = None,
        operation_result: KnowledgeOperationResultV1 | None = None,
        message_code: str,
    ) -> KnowledgeAdministrationResultV1:
        return KnowledgeAdministrationResultV1(
            status=status,
            intent=intent,
            item=item,
            inventory=inventory,
            candidates=candidates,
            confirmation_required=confirmation_required,
            confirmation_token=confirmation_token,
            operation_result=operation_result,
            message_code=message_code,
        )

    @staticmethod
    def _error_result(
        intent: KnowledgeAdministrationIntentV1,
        *,
        status: KnowledgeAdministrationStatusV1,
        inventory: KnowledgeInventoryV1 | None = None,
        item: KnowledgeInventoryItemV1 | None = None,
        message_code: str,
    ) -> KnowledgeAdministrationResultV1:
        return KnowledgeAdministrationResultV1(
            status=status,
            intent=intent,
            inventory=inventory,
            item=item,
            message_code=message_code,
        )


class _TargetResolution:
    def __init__(
        self,
        matches: tuple[KnowledgeInventoryItemV1, ...],
        *,
        ambiguous: bool = False,
    ) -> None:
        self.matches = matches
        self.ambiguous = ambiguous

    @property
    def item(self) -> KnowledgeInventoryItemV1 | None:
        return self.matches[0] if len(self.matches) == 1 and not self.ambiguous else None

    @classmethod
    def from_matches(
        cls,
        matches: tuple[KnowledgeInventoryItemV1, ...],
    ) -> _TargetResolution:
        return cls(matches, ambiguous=len(matches) > 1)

    @classmethod
    def not_found(cls) -> _TargetResolution:
        return cls(())


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def _inventory_with_items(
    inventory: KnowledgeInventoryV1,
    items: tuple[KnowledgeInventoryItemV1, ...],
) -> KnowledgeInventoryV1:
    summary = KnowledgeInventorySummaryV1(
        total=len(items),
        indexed=sum(item.mode is KnowledgeAccessModeV1.INDEXED for item in items),
        live=sum(item.mode is KnowledgeAccessModeV1.LIVE for item in items),
        active=sum(item.lifecycle_state == "active" for item in items),
        disabled=sum(item.lifecycle_state == "disabled" for item in items),
        attention_required=sum(
            item.lifecycle_state in {"error", "detach_blocked"}
            or (
                item.runtime_available is False
                and item.enabled
                and not item.detached
            )
            for item in items
        ),
    )
    return KnowledgeInventoryV1(
        tenant_id=inventory.tenant_id,
        workspace_id=inventory.workspace_id,
        items=items,
        summary=summary,
        updated_at=inventory.updated_at,
    )


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
