# © Artur Czarnecki. All rights reserved.

"""Provider-neutral connected source models for LKW."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RemoteResourceTypeV1(StrEnum):
    SLACK_CONVERSATION = "slack_conversation"
    MSGRAPH_TEAMS_CHAT = "teams_chat"
    MSGRAPH_MAIL_FOLDER = "mail_folder"
    MSGRAPH_TEAMS_CHANNEL = "teams_channel"
    MSGRAPH_CALENDAR = "calendar"
    GOOGLE_WORKSPACE_CALENDAR = "google_workspace_calendar"
    GOOGLE_WORKSPACE_DOCS = "google_workspace_docs"
    GOOGLE_WORKSPACE_SHEETS = "google_workspace_sheets"


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
    remote_resource_id: str | None = Field(default=None, min_length=1, max_length=256)
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
    ABORTED = "aborted"


ConnectedSourceDeliveryStatus = ConnectedSourceDeliveryStatusV1

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_SEQUENCE_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")


def _validate_sequence_identity(value: str, field_name: str) -> str:
    if value != value.strip() or _SEQUENCE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name}_must_be_normalized")
    return value


class ConnectedSourceReconciliationStateV1(StrEnum):
    NEW_RECONCILIATION = "new_reconciliation"
    CONTINUATION = "continuation"


ConnectedSourceReconciliationState = ConnectedSourceReconciliationStateV1


class ConnectedSourceDeliveryPublicationStateV1(StrEnum):
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"


ConnectedSourceDeliveryPublicationState = ConnectedSourceDeliveryPublicationStateV1


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
    payload_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    publication_state: ConnectedSourceDeliveryPublicationStateV1 | None = None
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

    @field_validator("payload_fingerprint")
    @classmethod
    def _validate_payload_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_HEX_RE.fullmatch(value):
            raise ValueError("connected_source_payload_fingerprint_invalid")
        return value

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> ConnectedSourceDeliveryReceiptV1:
        if self.status is ConnectedSourceDeliveryStatusV1.IN_PROGRESS:
            if self.completed_at is not None:
                raise ValueError("connected_source_delivery_in_progress_completed_at_set")
            if self.items_failed != 0:
                raise ValueError("connected_source_delivery_in_progress_items_failed")
            if self.publication_state not in {
                None,
                ConnectedSourceDeliveryPublicationStateV1.PREPARING,
                ConnectedSourceDeliveryPublicationStateV1.PREPARED,
            }:
                raise ValueError("connected_source_delivery_in_progress_state_invalid")
        elif self.status is ConnectedSourceDeliveryStatusV1.COMPLETED:
            if self.completed_at is None:
                raise ValueError("connected_source_delivery_completed_missing_completed_at")
            if self.items_failed != 0:
                raise ValueError("connected_source_delivery_completed_items_failed")
            if self.publication_state not in {
                None,
                ConnectedSourceDeliveryPublicationStateV1.COMMITTED,
            }:
                raise ValueError("connected_source_delivery_completed_state_invalid")
        elif self.status is ConnectedSourceDeliveryStatusV1.ABORTED:
            if self.completed_at is None:
                raise ValueError("connected_source_delivery_aborted_missing_completed_at")
            if self.publication_state not in {
                None,
                ConnectedSourceDeliveryPublicationStateV1.ABORTED,
            }:
                raise ValueError("connected_source_delivery_aborted_state_invalid")
        return self


ConnectedSourceDeliveryReceipt = ConnectedSourceDeliveryReceiptV1


class ConnectedSourceDeliverySequenceHeadV1(BaseModel):
    """Constant-size CAS-updated sequence authority for one binding stream."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    next_sequence: int = Field(..., ge=1)

    _validate_identity = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
    )(
        lambda value, info: _validate_sequence_identity(
            value, info.field_name or "identity"
        )
    )


ConnectedSourceDeliverySequenceHead = ConnectedSourceDeliverySequenceHeadV1


class ConnectedSourceDeliverySequenceAssignmentV1(BaseModel):
    """Immutable sequence reservation for exactly one delivery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    delivery_id: str
    materialization_sequence: int = Field(..., gt=0)
    assigned_at: datetime

    _validate_identity = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
    )(
        lambda value, info: _validate_sequence_identity(
            value, info.field_name or "identity"
        )
    )

    @field_validator("delivery_id")
    @classmethod
    def _validate_delivery_id(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("connected_source_delivery_id_invalid")
        return value

    @field_validator("assigned_at")
    @classmethod
    def _validate_assigned_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("connected_source_sequence_assigned_at_timezone_aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("connected_source_sequence_assigned_at_must_be_utc")
        return value.astimezone(UTC)


ConnectedSourceDeliverySequenceAssignment = ConnectedSourceDeliverySequenceAssignmentV1


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
    workspace_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_id: str | None = Field(default=None, min_length=1, max_length=128)
    indexed_source_binding_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    knowledge_source_binding_ref: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    operation_id: str = Field(..., min_length=1, max_length=128)
    delivery_id: str = Field(..., min_length=64, max_length=64)
    documents_indexed: int = Field(..., ge=0)
    documents_unchanged: int = Field(..., ge=0)
    items_failed: int = Field(..., ge=0)
    accounted_at: datetime
    ownership_classification: str = "LEGACY_MIGRATION_REQUIRED"

    @field_validator("delivery_id")
    @classmethod
    def _validate_delivery_id(cls, value: str) -> str:
        if not _SHA256_HEX_RE.fullmatch(value):
            raise ValueError("connected_source_delivery_id_invalid")
        return value

    @model_validator(mode="after")
    def _validate_ownership(self) -> ConnectedSourceOperationDeliveryAccountingV1:
        identity = (
            self.workspace_id,
            self.source_id,
            self.indexed_source_binding_id,
            self.knowledge_source_binding_ref,
        )
        if any(value is not None for value in identity) and any(
            value is None for value in identity
        ):
            raise ValueError("connected_source_accounting_ownership_incomplete")
        if self.ownership_classification == "COMPLETE_OWNERSHIP" and any(
            value is None for value in identity
        ):
            raise ValueError("connected_source_accounting_ownership_incomplete")
        if (
            self.ownership_classification != "COMPLETE_OWNERSHIP"
            and all(value is not None for value in identity)
        ):
            raise ValueError("connected_source_accounting_classification_mismatch")
        return self


ConnectedSourceOperationDeliveryAccounting = ConnectedSourceOperationDeliveryAccountingV1


class ConnectedSourceSyncEnqueueIntentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    indexed_source_binding_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    knowledge_source_binding_ref: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    operation_id: str = Field(..., min_length=1, max_length=128)
    enqueue_generation: int = Field(..., ge=1)
    last_enqueued_generation: int = Field(default=0, ge=0)
    last_task_id: str | None = None
    last_queue_provider: str | None = None
    updated_at: datetime
    ownership_classification: str = "LEGACY_MIGRATION_REQUIRED"

    @model_validator(mode="after")
    def _validate_enqueue_generation_invariants(self) -> ConnectedSourceSyncEnqueueIntentV1:
        if self.last_enqueued_generation > self.enqueue_generation:
            raise ValueError("connected_source_enqueue_generation_invariant_violation")
        identity = (
            self.indexed_source_binding_id,
            self.knowledge_source_binding_ref,
        )
        if any(value is not None for value in identity) and any(
            value is None for value in identity
        ):
            raise ValueError("connected_source_enqueue_ownership_incomplete")
        if self.ownership_classification == "COMPLETE_OWNERSHIP" and any(
            value is None for value in identity
        ):
            raise ValueError("connected_source_enqueue_ownership_incomplete")
        if (
            self.ownership_classification != "COMPLETE_OWNERSHIP"
            and all(value is not None for value in identity)
        ):
            raise ValueError("connected_source_enqueue_classification_mismatch")
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
