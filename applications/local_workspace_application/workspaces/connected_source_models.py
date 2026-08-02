# © Artur Czarnecki. All rights reserved.

"""Provider-neutral connected source models for LKW."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import re

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
