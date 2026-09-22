# © Artur Czarnecki. All rights reserved.

"""Immutable suspended execution operation descriptor (UCA-6C-R6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
)
from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.contracts.validation import validate_content_digest

SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1 = (
    "suspended_execution_operation_descriptor.v1"
)


class SuspendedOperationMaterializationState(StrEnum):
    """Materialization lifecycle — not continuation lifecycle."""

    PREPARED = "prepared"
    BLOCKED = "blocked"
    CLAIMED = "claimed"
    CONSUMED = "consumed"
    ABANDONED = "abandoned"


class SuspendedExecutionOperationDescriptor(BaseModel):
    """Durable work description for exact catalog/tool re-entry after HITL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1] = (
        SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1
    )
    suspended_operation_id: str = Field(min_length=1)
    operation_kind: SuspendedOperationKind
    identity: ExecutionContinuationIdentity
    continuation_id: str = Field(min_length=1)
    invocation_scope_id: str = Field(min_length=1)
    materialization_state: SuspendedOperationMaterializationState
    materialization_revision: int = Field(ge=0)
    claim_ownership: LeaseOwnership | None = None
    payload_digest: str = Field(min_length=1)
    payload: SerializedSuspendedOperationEnvelope

    @field_validator("payload_digest")
    @classmethod
    def _validate_payload_digest(cls, value: str) -> str:
        return validate_content_digest(value)

    @field_validator("suspended_operation_id", "continuation_id", "invocation_scope_id")
    @classmethod
    def _strip_ids(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must be non-empty")
        return normalized


__all__ = [
    "SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1",
    "SuspendedExecutionOperationDescriptor",
    "SuspendedOperationMaterializationState",
]
