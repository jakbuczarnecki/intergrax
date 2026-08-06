# © Artur Czarnecki. All rights reserved.

"""Provider-neutral connected source models for LKW."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RemoteResourceTypeV1(StrEnum):
    SLACK_CONVERSATION = "slack_conversation"


class SlackConversationKindV1(StrEnum):
    PUBLIC_CHANNEL = "public_channel"
    PRIVATE_CHANNEL = "private_channel"
    IM = "im"
    MPIM = "mpim"


class RemoteResourceCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opaque_candidate_ref: str = Field(..., min_length=1, max_length=1024)
    resource_type: RemoteResourceTypeV1
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    conversation_kind: SlackConversationKindV1 | None = None
    is_archived: bool = False
    is_private: bool = False
    safe_description: str | None = Field(default=None, max_length=512)


class RemoteResourceDiscoveryPageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[RemoteResourceCandidateV1, ...] = ()
    next_cursor: str | None = None


class ConnectedSourceDeliveryStatusV1(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


ConnectedSourceDeliveryStatus = ConnectedSourceDeliveryStatusV1

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


class ConnectedSourceReconciliationStateV1(StrEnum):
    NEW_RECONCILIATION = "new_reconciliation"
    CONTINUATION = "continuation"


ConnectedSourceReconciliationState = ConnectedSourceReconciliationStateV1


class ConnectedSourceDeliveryReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    knowledge_source_binding_ref: str = Field(..., min_length=1, max_length=128)
    delivery_id: str = Field(..., min_length=64, max_length=64)
    binding_configuration_version: int = Field(..., ge=1)
    # Optional only for parsing historical receipts; new delivery paths must set it.
    materialization_sequence: int | None = Field(default=None, gt=0)
    operation_id: str = Field(..., min_length=1, max_length=128)
    status: ConnectedSourceDeliveryStatusV1
    documents_indexed: int = Field(default=0, ge=0)
    documents_unchanged: int = Field(default=0, ge=0)
    items_failed: int = Field(default=0, ge=0)
    created_at: datetime
    completed_at: datetime | None = None

    @field_validator("delivery_id")
    @classmethod
    def _validate_delivery_id(cls, value: str) -> str:
        if not _SHA256_HEX_RE.fullmatch(value):
            raise ValueError("connected_source_delivery_id_invalid")
        return value

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> ConnectedSourceDeliveryReceiptV1:
        if self.status is ConnectedSourceDeliveryStatusV1.IN_PROGRESS:
            if self.completed_at is not None:
                raise ValueError("connected_source_delivery_in_progress_completed_at_set")
            if self.items_failed != 0:
                raise ValueError("connected_source_delivery_in_progress_items_failed")
        elif self.status is ConnectedSourceDeliveryStatusV1.COMPLETED:
            if self.completed_at is None:
                raise ValueError("connected_source_delivery_completed_missing_completed_at")
            if self.items_failed != 0:
                raise ValueError("connected_source_delivery_completed_items_failed")
        return self


ConnectedSourceDeliveryReceipt = ConnectedSourceDeliveryReceiptV1


class ConnectedSourceDeliverySequenceLedgerV1(BaseModel):
    """CAS-updated delivery sequence authority for one binding stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    next_sequence: int = Field(..., ge=1)
    delivery_sequences: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence_invariants(
        self,
    ) -> ConnectedSourceDeliverySequenceLedgerV1:
        values = tuple(self.delivery_sequences.values())
        if (
            len(set(values)) != len(values)
            or any(sequence < 1 for sequence in values)
            or (values and self.next_sequence <= max(values))
        ):
            raise ValueError("connected_source_delivery_sequence_ledger_invalid")
        return self


ConnectedSourceDeliverySequenceLedger = ConnectedSourceDeliverySequenceLedgerV1


class ConnectedSourceReadinessStateV1(StrEnum):
    DISABLED = "disabled"
    SIGNING_KEY_MISSING = "signing_key_missing"
    SLACK_INTEGRATION_UNAVAILABLE = "slack_integration_unavailable"
    MAPPING_INCOMPLETE = "mapping_incomplete"
    READY = "ready"


ConnectedSourceReadinessState = ConnectedSourceReadinessStateV1


class ConnectedSourceOperationDeliveryAccountingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    operation_id: str = Field(..., min_length=1, max_length=128)
    delivery_id: str = Field(..., min_length=64, max_length=64)
    documents_indexed: int = Field(..., ge=0)
    documents_unchanged: int = Field(..., ge=0)
    items_failed: int = Field(..., ge=0)
    accounted_at: datetime

    @field_validator("delivery_id")
    @classmethod
    def _validate_delivery_id(cls, value: str) -> str:
        if not _SHA256_HEX_RE.fullmatch(value):
            raise ValueError("connected_source_delivery_id_invalid")
        return value


ConnectedSourceOperationDeliveryAccounting = ConnectedSourceOperationDeliveryAccountingV1


class ConnectedSourceSyncEnqueueIntentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    operation_id: str = Field(..., min_length=1, max_length=128)
    enqueue_generation: int = Field(..., ge=1)
    last_enqueued_generation: int = Field(default=0, ge=0)
    last_task_id: str | None = None
    last_queue_provider: str | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def _validate_enqueue_generation_invariants(self) -> ConnectedSourceSyncEnqueueIntentV1:
        if self.last_enqueued_generation > self.enqueue_generation:
            raise ValueError("connected_source_enqueue_generation_invariant_violation")
        return self


ConnectedSourceSyncEnqueueIntent = ConnectedSourceSyncEnqueueIntentV1


class ConnectedSourceDiscoveryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class ConnectedSourceBindingError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class ConnectedSourceSyncSinkError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code
