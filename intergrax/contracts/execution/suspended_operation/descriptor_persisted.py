# © Artur Czarnecki. All rights reserved.

"""Persisted read boundary for suspended execution operation descriptors (COMPAT-X-R2-R2)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_governance_hitl import LogicalInvocationFingerprint
from intergrax.contracts.execution.suspended_operation.authority_scope_compat import (
    UnknownInvocationScopeError,
    recognize_authority_scope_from_invocation,
)
from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1,
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.contracts.validation import validate_content_digest

LEGACY_DESCRIPTOR_WITHOUT_AUTHORITY_SCOPE = "LEGACY_DESCRIPTOR_WITHOUT_AUTHORITY_SCOPE"


class _LegacySuspendedExecutionOperationDescriptor(BaseModel):
    """Historical durable descriptor inside v1 envelope — authority_scope omitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1
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

    @field_validator("payload_digest")
    @classmethod
    def _validate_payload_digest(cls, value: str) -> str:
        return validate_content_digest(value)


def _is_legacy_descriptor_mapping(data: Mapping[str, object]) -> bool:
    if data.get("authority_scope") is not None:
        return False
    if "invocation_scope_id" not in data:
        return False
    schema_version = data.get("schema_version")
    if schema_version is not None and schema_version != SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1:
        return False
    return True


def parse_suspended_execution_operation_descriptor(
    data: object,
) -> SuspendedExecutionOperationDescriptor:
    if not isinstance(data, dict):
        raise ValueError("suspended operation descriptor must be a mapping")
    mapping: Mapping[str, object] = data
    schema_version = mapping.get("schema_version")
    if schema_version == SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1:
        if mapping.get("authority_scope") is not None:
            return SuspendedExecutionOperationDescriptor.model_validate(data)
        if not _is_legacy_descriptor_mapping(mapping):
            raise ValueError("unsupported suspended operation descriptor shape")
        return _migrate_legacy_descriptor(mapping)
    if schema_version is None and _is_legacy_descriptor_mapping(mapping):
        return _migrate_legacy_descriptor(mapping)
    raise ValueError(
        f"unsupported suspended operation descriptor schema version {schema_version!r}; "
        f"expected {SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1!r}"
    )


def _migrate_legacy_descriptor(data: Mapping[str, object]) -> SuspendedExecutionOperationDescriptor:
    legacy = _LegacySuspendedExecutionOperationDescriptor.model_validate(dict(data))
    scope = legacy.invocation_scope_id
    try:
        authority_scope = recognize_authority_scope_from_invocation(scope)
    except UnknownInvocationScopeError as exc:
        raise ValueError("unknown invocation scope for legacy descriptor") from exc
    return SuspendedExecutionOperationDescriptor(
        schema_version=SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1,
        suspended_operation_id=legacy.suspended_operation_id,
        operation_kind=legacy.operation_kind,
        identity=legacy.identity,
        continuation_id=legacy.continuation_id,
        invocation_scope_id=legacy.invocation_scope_id,
        materialization_state=legacy.materialization_state,
        materialization_revision=legacy.materialization_revision,
        claim_ownership=legacy.claim_ownership,
        payload_digest=legacy.payload_digest,
        payload=legacy.payload,
        pause_generation=legacy.pause_generation,
        logical_invocation_fingerprint=legacy.logical_invocation_fingerprint,
        authority_scope=authority_scope,
    )


__all__ = [
    "LEGACY_DESCRIPTOR_WITHOUT_AUTHORITY_SCOPE",
    "parse_suspended_execution_operation_descriptor",
]
