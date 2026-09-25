# © Artur Czarnecki. All rights reserved.

"""Immutable suspended execution operation descriptor (UCA-6C-R6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.authority_scope_compat import (
    UnknownInvocationScopeError,
    invocation_scope_matches_authority_scope,
    recognize_authority_scope_from_invocation,
)
from intergrax.contracts.agent_governance_hitl import LogicalInvocationFingerprint
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
    pause_generation: int = Field(default=1, ge=1)
    logical_invocation_fingerprint: LogicalInvocationFingerprint | None = None
    authority_scope: SuspendedOperationAuthorityScope

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_authority_scope(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if data.get("authority_scope") is not None:
            return data
        scope = data.get("invocation_scope_id")
        if not isinstance(scope, str):
            raise ValueError("invocation_scope_id required for legacy descriptor")
        try:
            data["authority_scope"] = recognize_authority_scope_from_invocation(
                scope,
            ).value
        except UnknownInvocationScopeError as exc:
            raise ValueError("unknown invocation scope for legacy descriptor") from exc
        return data

    @model_validator(mode="after")
    def _authority_scope_matches_invocation(
        self,
    ) -> SuspendedExecutionOperationDescriptor:
        try:
            recognize_authority_scope_from_invocation(self.invocation_scope_id)
        except UnknownInvocationScopeError as exc:
            raise ValueError("unknown invocation scope") from exc
        if not invocation_scope_matches_authority_scope(
            self.invocation_scope_id,
            self.authority_scope,
        ):
            raise ValueError("authority scope incompatible with invocation scope")
        return self

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
