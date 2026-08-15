# © Artur Czarnecki. All rights reserved.

"""Durable, safe idempotency receipts for conversational interaction events."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_PARTITION = "lkw.conversation_interaction:event_receipt"
_TTL_SECONDS = 7 * 24 * 60 * 60
_MAX_RESPONSE_LENGTH = 4_000


class ConversationEventReceiptStatus(StrEnum):
    PROCESSING = "processing"
    RESPONSE_PENDING = "response_pending"
    RESPONSE_FAILED = "response_failed"
    RESPONSE_SENT = "response_sent"


class ConversationEventMemoryStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class ConversationEventReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    conversation_connection_ref: str = Field(min_length=1, max_length=128)
    provider_event_ref: str = Field(min_length=1, max_length=256)
    status: ConversationEventReceiptStatus
    execution_id: str = Field(min_length=1, max_length=128)
    safe_response: str | None = Field(default=None, max_length=_MAX_RESPONSE_LENGTH)
    response_hash: str | None = Field(default=None, max_length=64)
    memory_status: ConversationEventMemoryStatus = (
        ConversationEventMemoryStatus.NOT_REQUIRED
    )
    memory_revision_id: str | None = Field(default=None, max_length=128)
    memory_error_code: str | None = Field(default=None, max_length=128)
    safe_user_memory_text: str | None = Field(default=None, max_length=16_000)
    created_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_identity_fields(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        for field_name in (
            "tenant_id",
            "conversation_connection_ref",
            "provider_event_ref",
            "execution_id",
        ):
            field_value = payload.get(field_name)
            if isinstance(field_value, str):
                payload[field_name] = field_value.strip()
        return payload

    @model_validator(mode="after")
    def _validate_invariants(self) -> "ConversationEventReceipt":
        identity_values = (
            self.tenant_id,
            self.conversation_connection_ref,
            self.provider_event_ref,
            self.execution_id,
        )
        if any(not value.strip() for value in identity_values):
            raise ValueError("receipt identity fields must be nonblank")
        if not _is_utc_timestamp(self.created_at) or (
            self.completed_at is not None
            and not _is_utc_timestamp(self.completed_at)
        ):
            raise ValueError("receipt timestamps must be timezone-aware UTC")

        if self.memory_status is ConversationEventMemoryStatus.COMPLETED:
            if self.memory_revision_id is None or self.memory_error_code is not None:
                raise ValueError("completed memory state is inconsistent")
        elif self.memory_status is ConversationEventMemoryStatus.FAILED:
            if not self.memory_error_code or self.memory_revision_id is not None:
                raise ValueError("failed memory state is inconsistent")
        elif self.memory_status is ConversationEventMemoryStatus.PENDING:
            if self.memory_revision_id is not None or self.memory_error_code is not None:
                raise ValueError("pending memory state is inconsistent")
            if not self.safe_user_memory_text or not self.safe_user_memory_text.strip():
                raise ValueError("pending memory requires safe_user_memory_text")
        elif (
            self.memory_revision_id is not None
            or self.memory_error_code is not None
        ):
            raise ValueError("not-required memory state is inconsistent")
        if self.memory_status is ConversationEventMemoryStatus.NOT_REQUIRED:
            if self.safe_user_memory_text is not None:
                raise ValueError("not-required memory cannot contain safe_user_memory_text")

        if self.status is ConversationEventReceiptStatus.PROCESSING:
            if (
                self.safe_response is not None
                or self.response_hash is not None
                or self.completed_at is not None
                or self.memory_status is not ConversationEventMemoryStatus.NOT_REQUIRED
            ):
                raise ValueError("processing receipt cannot contain a response")
            return self

        if (
            not self.safe_response
            or not self.safe_response.strip()
            or self.response_hash
            != hashlib.sha256(self.safe_response.encode("utf-8")).hexdigest()
            or self.completed_at is None
        ):
            raise ValueError("completed receipt response fields are inconsistent")
        if (
            self.status is ConversationEventReceiptStatus.RESPONSE_SENT
            and self.memory_status is ConversationEventMemoryStatus.PENDING
        ):
            raise ValueError("sent receipt memory must be terminal")
        return self


class ConversationEventReceiptError(RuntimeError):
    """Stable receipt-storage failure without storage/provider details."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class ConversationEventReceiptClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    owned: bool
    receipt: ConversationEventReceipt


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _is_utc_timestamp(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


def _row_key(*, tenant_id: str, connection_ref: str, event_ref: str) -> str:
    canonical = "\x1f".join(
        (tenant_id.strip(), connection_ref.strip(), event_ref.strip())
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record(receipt: ConversationEventReceipt) -> DocumentRecord:
    return DocumentRecord(
        partition_key=_PARTITION,
        row_key=_row_key(
            tenant_id=receipt.tenant_id,
            connection_ref=receipt.conversation_connection_ref,
            event_ref=receipt.provider_event_ref,
        ),
        data=receipt.model_dump(mode="json"),
        ttl_seconds=_TTL_SECONDS,
    )


class ConversationInteractionEventReceiptRepository:
    """CAS-backed event receipt repository over the shared DocumentStore."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    def _conditional_store(self) -> ConditionalDocumentStore:
        if not isinstance(self._store, ConditionalDocumentStore):
            raise ConversationEventReceiptError(
                "conversation_receipt_conditional_store_required"
            )
        return self._store

    def _get(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
        provider_event_ref: str,
    ) -> ConversationEventReceipt | None:
        stored = self._store.get(
            _PARTITION,
            _row_key(
                tenant_id=tenant_id,
                connection_ref=conversation_connection_ref,
                event_ref=provider_event_ref,
            ),
        )
        if stored is None:
            return None
        try:
            return ConversationEventReceipt.model_validate(dict(stored.data))
        except Exception as exc:  # noqa: BLE001 - safe storage boundary
            raise ConversationEventReceiptError(
                "conversation_receipt_malformed"
            ) from exc

    def claim(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
        provider_event_ref: str,
        execution_id: str,
    ) -> ConversationEventReceiptClaim:
        now = _utcnow()
        candidate = ConversationEventReceipt(
            tenant_id=tenant_id.strip(),
            conversation_connection_ref=conversation_connection_ref.strip(),
            provider_event_ref=provider_event_ref.strip(),
            status=ConversationEventReceiptStatus.PROCESSING,
            execution_id=execution_id.strip(),
            created_at=now,
        )
        if self._conditional_store().put_if_absent(_record(candidate)):
            return ConversationEventReceiptClaim(owned=True, receipt=candidate)
        existing = self._get(
            tenant_id=tenant_id,
            conversation_connection_ref=conversation_connection_ref,
            provider_event_ref=provider_event_ref,
        )
        if existing is None:
            raise ConversationEventReceiptError("conversation_receipt_claim_conflict")
        return ConversationEventReceiptClaim(owned=False, receipt=existing)

    def _replace(
        self,
        current: ConversationEventReceipt,
        replacement: ConversationEventReceipt,
    ) -> None:
        if not self._conditional_store().replace_if_match(
            expected=_record(current),
            replacement=_record(replacement),
        ):
            raise ConversationEventReceiptError("conversation_receipt_update_conflict")

    @staticmethod
    def _transition(
        receipt: ConversationEventReceipt,
        *,
        update: dict[str, object],
    ) -> ConversationEventReceipt:
        payload = receipt.model_dump(mode="python")
        payload.update(update)
        return ConversationEventReceipt.model_validate(payload)

    @staticmethod
    def _ensure_transition(
        receipt: ConversationEventReceipt,
        target: ConversationEventReceiptStatus,
    ) -> None:
        allowed = {
            ConversationEventReceiptStatus.PROCESSING: {
                ConversationEventReceiptStatus.RESPONSE_PENDING
            },
            ConversationEventReceiptStatus.RESPONSE_PENDING: {
                ConversationEventReceiptStatus.RESPONSE_SENT,
                ConversationEventReceiptStatus.RESPONSE_FAILED,
            },
            ConversationEventReceiptStatus.RESPONSE_FAILED: {
                ConversationEventReceiptStatus.RESPONSE_SENT,
                ConversationEventReceiptStatus.RESPONSE_FAILED,
            },
            ConversationEventReceiptStatus.RESPONSE_SENT: set(),
        }
        if target not in allowed[receipt.status]:
            raise ConversationEventReceiptError(
                "conversation_receipt_transition_invalid"
            )

    def mark_response_pending(
        self,
        *,
        receipt: ConversationEventReceipt,
        response: str,
        memory_required: bool = False,
        safe_user_memory_text: str | None = None,
    ) -> ConversationEventReceipt:
        self._ensure_transition(
            receipt,
            ConversationEventReceiptStatus.RESPONSE_PENDING,
        )
        safe_response = response.strip()
        if not safe_response:
            raise ConversationEventReceiptError(
                "conversation_receipt_response_empty"
            )
        safe_response = safe_response[:_MAX_RESPONSE_LENGTH]
        memory_text: str | None = None
        if memory_required:
            if safe_user_memory_text is None or not safe_user_memory_text.strip():
                raise ConversationEventReceiptError(
                    "conversation_receipt_memory_text_empty"
                )
            memory_text = safe_user_memory_text.strip()[:16_000]
        replacement = self._transition(
            receipt,
            update={
                "status": ConversationEventReceiptStatus.RESPONSE_PENDING,
                "safe_response": safe_response,
                "response_hash": hashlib.sha256(
                    safe_response.encode("utf-8")
                ).hexdigest(),
                "completed_at": _utcnow(),
                "memory_status": (
                    ConversationEventMemoryStatus.PENDING
                    if memory_required
                    else ConversationEventMemoryStatus.NOT_REQUIRED
                ),
                "memory_revision_id": None,
                "memory_error_code": None,
                "safe_user_memory_text": memory_text,
            },
        )
        self._replace(receipt, replacement)
        return replacement

    def mark_memory_completed(
        self,
        *,
        receipt: ConversationEventReceipt,
        revision_id: str,
    ) -> ConversationEventReceipt:
        if receipt.status not in {
            ConversationEventReceiptStatus.RESPONSE_PENDING,
            ConversationEventReceiptStatus.RESPONSE_FAILED,
        }:
            raise ConversationEventReceiptError(
                "conversation_receipt_memory_transition_invalid"
            )
        revision = revision_id.strip()
        if not revision:
            raise ConversationEventReceiptError(
                "conversation_receipt_memory_revision_empty"
            )
        replacement = self._transition(
            receipt,
            update={
                "memory_status": ConversationEventMemoryStatus.COMPLETED,
                "memory_revision_id": revision[:128],
                "memory_error_code": None,
            },
        )
        self._replace(receipt, replacement)
        return replacement

    def mark_memory_failed(
        self,
        *,
        receipt: ConversationEventReceipt,
        error_code: str,
    ) -> ConversationEventReceipt:
        if receipt.status not in {
            ConversationEventReceiptStatus.RESPONSE_PENDING,
            ConversationEventReceiptStatus.RESPONSE_FAILED,
        }:
            raise ConversationEventReceiptError(
                "conversation_receipt_memory_transition_invalid"
            )
        normalized = error_code.strip()
        if not normalized:
            raise ConversationEventReceiptError(
                "conversation_receipt_memory_error_empty"
            )
        replacement = self._transition(
            receipt,
            update={
                "memory_status": ConversationEventMemoryStatus.FAILED,
                "memory_revision_id": None,
                "memory_error_code": normalized[:128],
            },
        )
        self._replace(receipt, replacement)
        return replacement

    def mark_response_sent(
        self,
        *,
        receipt: ConversationEventReceipt,
    ) -> ConversationEventReceipt:
        self._ensure_transition(
            receipt,
            ConversationEventReceiptStatus.RESPONSE_SENT,
        )
        if receipt.memory_status is ConversationEventMemoryStatus.PENDING:
            raise ConversationEventReceiptError(
                "conversation_receipt_memory_not_terminal"
            )
        replacement = self._transition(
            receipt,
            update={"status": ConversationEventReceiptStatus.RESPONSE_SENT},
        )
        self._replace(receipt, replacement)
        return replacement

    def mark_response_failed(
        self,
        *,
        receipt: ConversationEventReceipt,
    ) -> ConversationEventReceipt:
        self._ensure_transition(
            receipt,
            ConversationEventReceiptStatus.RESPONSE_FAILED,
        )
        replacement = self._transition(
            receipt,
            update={"status": ConversationEventReceiptStatus.RESPONSE_FAILED},
        )
        self._replace(receipt, replacement)
        return replacement


__all__ = [
    "ConversationEventReceipt",
    "ConversationEventReceiptClaim",
    "ConversationEventReceiptError",
    "ConversationEventMemoryStatus",
    "ConversationEventReceiptStatus",
    "ConversationInteractionEventReceiptRepository",
]
