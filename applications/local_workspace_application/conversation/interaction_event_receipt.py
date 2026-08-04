# © Artur Czarnecki. All rights reserved.

"""Durable, safe idempotency receipts for conversational interaction events."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

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


class ConversationEventReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    conversation_connection_ref: str = Field(min_length=1, max_length=128)
    provider_event_ref: str = Field(min_length=1, max_length=256)
    status: ConversationEventReceiptStatus
    execution_id: str = Field(min_length=1, max_length=128)
    safe_response: str | None = Field(default=None, max_length=_MAX_RESPONSE_LENGTH)
    response_hash: str | None = Field(default=None, max_length=64)
    created_at: datetime
    completed_at: datetime | None = None


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

    def mark_response_pending(
        self,
        *,
        receipt: ConversationEventReceipt,
        response: str,
    ) -> ConversationEventReceipt:
        safe_response = response.strip()[:_MAX_RESPONSE_LENGTH]
        replacement = receipt.model_copy(
            update={
                "status": ConversationEventReceiptStatus.RESPONSE_PENDING,
                "safe_response": safe_response,
                "response_hash": hashlib.sha256(
                    safe_response.encode("utf-8")
                ).hexdigest(),
                "completed_at": _utcnow(),
            }
        )
        self._replace(receipt, replacement)
        return replacement

    def mark_response_sent(self, *, receipt: ConversationEventReceipt) -> None:
        self._replace(
            receipt,
            receipt.model_copy(
                update={"status": ConversationEventReceiptStatus.RESPONSE_SENT}
            ),
        )

    def mark_response_failed(self, *, receipt: ConversationEventReceipt) -> None:
        self._replace(
            receipt,
            receipt.model_copy(
                update={"status": ConversationEventReceiptStatus.RESPONSE_FAILED}
            ),
        )


__all__ = [
    "ConversationEventReceipt",
    "ConversationEventReceiptClaim",
    "ConversationEventReceiptError",
    "ConversationEventReceiptStatus",
    "ConversationInteractionEventReceiptRepository",
]
