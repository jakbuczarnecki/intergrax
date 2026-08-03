# © Artur Czarnecki. All rights reserved.

"""Workspace Knowledge Configuration domain models (LKW-KNOWLEDGE-ACCESS-1B-1)."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory

_CONNECTION_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_connection_ref(value: str) -> str:
    if _CONNECTION_REF_RE.fullmatch(value) is None:
        raise ValueError("connection_ref_invalid")
    return value


def _validate_sha256_hex(value: str) -> str:
    if _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError("sha256_hex_invalid")
    return value


def _canonicalize_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        raise ValueError("string_tuple_null_forbidden")
    if isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError("string_tuple_invalid")
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError("string_tuple_invalid")
        trimmed = item.strip()
        if not trimmed:
            raise ValueError("string_tuple_blank_value")
        if trimmed not in seen:
            seen.add(trimmed)
            result.append(trimmed)
    result.sort()
    return tuple(result)


class IndexedSourceSyncModeV1(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"


class KnowledgeAudienceEligibilityV1(StrEnum):
    PERSONAL_ONLY = "personal_only"
    SHARED_ALLOWED = "shared_allowed"


IndexedSourceAudienceEligibilityV1 = KnowledgeAudienceEligibilityV1


class WorkspaceIndexedSourceBindingStatusV1(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class LiveAccessBindingStatusV1(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    REVOKED = "revoked"


class QueryPolicyModeV1(StrEnum):
    INDEXED_ONLY = "indexed_only"
    LIVE_ONLY = "live_only"


class LiveResultRetentionV1(StrEnum):
    EPHEMERAL = "ephemeral"
    RECEIPT_ONLY = "receipt_only"


class WorkspaceConnectionAttachmentStatusV1(StrEnum):
    ATTACHED = "attached"
    UNAVAILABLE = "unavailable"
    DETACHED = "detached"


class WorkspaceKnowledgeMutationOperationV1(StrEnum):
    ATTACH_CONNECTION = "attach_connection"
    DETACH_CONNECTION = "detach_connection"
    CREATE_INDEXED_SOURCE = "create_indexed_source"
    DISABLE_INDEXED_SOURCE = "disable_indexed_source"
    CREATE_LIVE_ACCESS_BINDING = "create_live_access_binding"
    DISABLE_LIVE_ACCESS_BINDING = "disable_live_access_binding"
    UPDATE_QUERY_POLICY = "update_query_policy"


class WorkspaceKnowledgeMutationStatusV1(StrEnum):
    RESERVED = "reserved"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    RECOVERY_REQUIRED = "recovery_required"


class WorkspaceKnowledgeMutationOutcomeV1(StrEnum):
    APPLIED = "applied"
    EXISTING_RESULT = "existing_result"


class WorkspaceConnectionAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    status: WorkspaceConnectionAttachmentStatusV1

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    created_at: datetime
    updated_at: datetime

    @field_validator("connection_ref")
    @classmethod
    def _validate_connection_ref_field(cls, value: str) -> str:
        return _validate_connection_ref(value)


class WorkspaceIndexedSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    knowledge_source_binding_ref: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)

    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.INCREMENTAL
    status: WorkspaceIndexedSourceBindingStatusV1 = (
        WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    )
    audience_eligibility: KnowledgeAudienceEligibilityV1 = (
        KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
    )

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    semantic_identity_hash: str = Field(..., min_length=64, max_length=64)

    created_at: datetime
    updated_at: datetime

    cached_safe_display_label: str | None = Field(default=None, max_length=256)

    @field_validator("semantic_identity_hash")
    @classmethod
    def _validate_semantic_identity_hash(cls, value: str) -> str:
        return _validate_sha256_hex(value)


class WorkspaceLiveAccessBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    allowed_capability_ids: tuple[str, ...] = Field(..., min_length=1)

    derived_provider_id: str = Field(..., min_length=1, max_length=64)
    derived_integration_kind: IntegrationCategory
    derived_resource_type: str | None = Field(default=None, max_length=64)
    derived_safe_display_label: str = Field(..., min_length=1, max_length=256)

    status: LiveAccessBindingStatusV1 = LiveAccessBindingStatusV1.ACTIVE
    audience_eligibility: KnowledgeAudienceEligibilityV1 = (
        KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
    )

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    semantic_identity_hash: str = Field(..., min_length=64, max_length=64)

    created_at: datetime
    updated_at: datetime

    @field_validator("connection_ref")
    @classmethod
    def _validate_connection_ref_field(cls, value: str) -> str:
        return _validate_connection_ref(value)

    @field_validator("semantic_identity_hash")
    @classmethod
    def _validate_semantic_identity_hash(cls, value: str) -> str:
        return _validate_sha256_hex(value)

    @field_validator("allowed_capability_ids", mode="before")
    @classmethod
    def _normalize_capability_ids(cls, value: Any) -> tuple[str, ...]:
        normalized = _canonicalize_string_tuple(value)
        if not normalized:
            raise ValueError("allowed_capability_ids_required")
        return normalized


class WorkspaceQueryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    mode: QueryPolicyModeV1 = QueryPolicyModeV1.INDEXED_ONLY

    allowed_connection_refs: tuple[str, ...] = ()
    allowed_capability_ids: tuple[str, ...] = ()

    max_live_calls: int = Field(default=0, ge=0, le=50)
    max_total_duration_ms: int = Field(default=30_000, ge=1, le=300_000)
    max_result_items: int = Field(default=50, ge=1, le=500)
    max_result_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)

    live_result_retention: LiveResultRetentionV1 = LiveResultRetentionV1.EPHEMERAL

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    updated_at: datetime

    @field_validator("allowed_connection_refs", mode="before")
    @classmethod
    def _normalize_connection_refs(cls, value: Any) -> tuple[str, ...]:
        return _canonicalize_string_tuple(value)

    @field_validator("allowed_capability_ids", mode="before")
    @classmethod
    def _normalize_capability_ids(cls, value: Any) -> tuple[str, ...]:
        return _canonicalize_string_tuple(value)

    @field_validator("allowed_connection_refs")
    @classmethod
    def _validate_connection_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            _validate_connection_ref(item)
        return value

    @model_validator(mode="after")
    def _validate_mode_invariants(self) -> Self:
        if self.mode is QueryPolicyModeV1.INDEXED_ONLY:
            if self.allowed_connection_refs:
                raise ValueError("indexed_only_forbids_connection_refs")
            if self.allowed_capability_ids:
                raise ValueError("indexed_only_forbids_capability_ids")
            if self.max_live_calls != 0:
                raise ValueError("indexed_only_forbids_live_calls")
            if self.live_result_retention is not LiveResultRetentionV1.EPHEMERAL:
                raise ValueError("indexed_only_requires_ephemeral_retention")
        elif self.mode is QueryPolicyModeV1.LIVE_ONLY:
            if not self.allowed_connection_refs:
                raise ValueError("live_only_requires_connection_refs")
            if not self.allowed_capability_ids:
                raise ValueError("live_only_requires_capability_ids")
            if self.max_live_calls < 1:
                raise ValueError("live_only_requires_live_calls")
        return self


class WorkspaceKnowledgeConfigurationHead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    committed_revision: int = Field(default=0, ge=0)

    pending_revision: int | None = Field(default=None, ge=1)
    pending_mutation_id: str | None = Field(default=None, max_length=128)

    last_committed_mutation_id: str | None = Field(default=None, max_length=128)

    updated_at: datetime

    @model_validator(mode="after")
    def _validate_head_state(self) -> Self:
        if self.pending_mutation_id is not None and not self.pending_mutation_id.strip():
            raise ValueError("pending_mutation_id_blank")
        if (
            self.last_committed_mutation_id is not None
            and not self.last_committed_mutation_id.strip()
        ):
            raise ValueError("last_committed_mutation_id_blank")

        has_pending_revision = self.pending_revision is not None
        has_pending_mutation = self.pending_mutation_id is not None

        if has_pending_revision != has_pending_mutation:
            raise ValueError("pending_head_fields_mismatched")

        if has_pending_revision:
            assert self.pending_revision is not None
            assert self.pending_mutation_id is not None
            if self.pending_revision != self.committed_revision + 1:
                raise ValueError("pending_revision_must_follow_committed")
            if not self.pending_mutation_id.strip():
                raise ValueError("pending_mutation_id_blank")
        else:
            if self.pending_mutation_id is not None:
                raise ValueError("idle_head_forbids_pending_mutation_id")

        return self


class WorkspaceKnowledgeMutationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mutation_id: str = Field(..., min_length=1, max_length=128)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    operation: WorkspaceKnowledgeMutationOperationV1

    idempotency_key_hash: str = Field(..., min_length=64, max_length=64)
    normalized_request_hash: str = Field(..., min_length=64, max_length=64)
    semantic_identity_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    stage_manifest_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    target_revision: int | None = Field(default=None, ge=1)
    stage_claim_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    committed_revision: int | None = Field(default=None, ge=0)

    status: WorkspaceKnowledgeMutationStatusV1
    outcome: WorkspaceKnowledgeMutationOutcomeV1 | None = None

    result_entity_type: str | None = Field(default=None, max_length=64)
    result_entity_id: str | None = Field(default=None, max_length=128)

    error_code: str | None = Field(default=None, max_length=128)

    created_at: datetime
    updated_at: datetime
    committed_at: datetime | None = None

    @field_validator(
        "idempotency_key_hash",
        "normalized_request_hash",
        "semantic_identity_hash",
        "stage_manifest_hash",
    )
    @classmethod
    def _validate_hashes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256_hex(value)

    @model_validator(mode="after")
    def _validate_mutation_state(self) -> Self:
        has_result_type = self.result_entity_type is not None
        has_result_id = self.result_entity_id is not None
        if has_result_type != has_result_id:
            raise ValueError("result_reference_fields_mismatched")

        status = self.status
        if status is WorkspaceKnowledgeMutationStatusV1.RESERVED:
            if self.committed_revision is not None:
                raise ValueError("reserved_forbids_committed_revision")
            if self.outcome is not None:
                raise ValueError("reserved_forbids_outcome")
            if self.committed_at is not None:
                raise ValueError("reserved_forbids_committed_at")
        elif status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
            if self.target_revision is None:
                raise ValueError("prepared_requires_target_revision")
            if self.committed_revision is not None:
                raise ValueError("prepared_forbids_committed_revision")
            if self.outcome is not None:
                raise ValueError("prepared_forbids_outcome")
            if self.committed_at is not None:
                raise ValueError("prepared_forbids_committed_at")
        elif status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
            if self.outcome is None:
                raise ValueError("committed_requires_outcome")
            if self.committed_at is None:
                raise ValueError("committed_requires_committed_at")
            if not has_result_type:
                raise ValueError("committed_requires_result_reference")
            if self.outcome is WorkspaceKnowledgeMutationOutcomeV1.APPLIED:
                if self.target_revision is None:
                    raise ValueError("applied_requires_target_revision")
                if self.committed_revision != self.target_revision:
                    raise ValueError("applied_revision_mismatch")
            elif self.outcome is WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT:
                if self.target_revision is not None:
                    raise ValueError("existing_result_forbids_target_revision")
                if self.committed_revision is None:
                    raise ValueError("existing_result_requires_committed_revision")
        elif status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
            if self.committed_revision is not None:
                raise ValueError("aborted_forbids_committed_revision")
            if self.outcome is not None:
                raise ValueError("aborted_forbids_outcome")
            if self.committed_at is not None:
                raise ValueError("aborted_forbids_committed_at")
        elif status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED:
            pass

        if self.stage_claim_id is not None:
            if self.target_revision is None:
                raise ValueError("stage_claim_requires_target_revision")
            if status not in (
                WorkspaceKnowledgeMutationStatusV1.RESERVED,
                WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
            ):
                raise ValueError("stage_claim_invalid_for_status")

        if status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED:
            if self.committed_at is not None:
                raise ValueError("non_committed_forbids_committed_at")

        return self


class WorkspaceKnowledgeConfigurationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    configuration_revision: int = Field(..., ge=0)

    connection_attachments: tuple[WorkspaceConnectionAttachment, ...] = ()
    indexed_sources: tuple[WorkspaceIndexedSourceBinding, ...] = ()
    live_access_bindings: tuple[WorkspaceLiveAccessBinding, ...] = ()
    query_policy: WorkspaceQueryPolicy | None = None

    updated_at: datetime

    @model_validator(mode="after")
    def _validate_projection(self) -> Self:
        if self.configuration_revision == 0:
            if (
                self.connection_attachments
                or self.indexed_sources
                or self.live_access_bindings
                or self.query_policy is not None
            ):
                raise ValueError("revision_zero_forbids_children")

        for attachment in self.connection_attachments:
            if attachment.tenant_id != self.tenant_id:
                raise ValueError("connection_attachment_tenant_mismatch")
            if attachment.workspace_id != self.workspace_id:
                raise ValueError("connection_attachment_workspace_mismatch")
            if attachment.effective_revision > self.configuration_revision:
                raise ValueError("connection_attachment_revision_too_new")

        for binding in self.indexed_sources:
            if binding.tenant_id != self.tenant_id:
                raise ValueError("indexed_source_tenant_mismatch")
            if binding.workspace_id != self.workspace_id:
                raise ValueError("indexed_source_workspace_mismatch")
            if binding.effective_revision > self.configuration_revision:
                raise ValueError("indexed_source_revision_too_new")

        for binding in self.live_access_bindings:
            if binding.tenant_id != self.tenant_id:
                raise ValueError("live_access_binding_tenant_mismatch")
            if binding.workspace_id != self.workspace_id:
                raise ValueError("live_access_binding_workspace_mismatch")
            if binding.effective_revision > self.configuration_revision:
                raise ValueError("live_access_binding_revision_too_new")

        if self.query_policy is not None:
            if self.query_policy.tenant_id != self.tenant_id:
                raise ValueError("query_policy_tenant_mismatch")
            if self.query_policy.workspace_id != self.workspace_id:
                raise ValueError("query_policy_workspace_mismatch")
            if self.query_policy.effective_revision > self.configuration_revision:
                raise ValueError("query_policy_revision_too_new")

        sorted_attachments = tuple(
            sorted(
                self.connection_attachments,
                key=lambda item: (item.connection_ref, item.attachment_id),
            )
        )
        sorted_indexed = tuple(
            sorted(self.indexed_sources, key=lambda item: item.indexed_source_binding_id)
        )
        sorted_live = tuple(
            sorted(
                self.live_access_bindings,
                key=lambda item: item.live_access_binding_id,
            )
        )

        if sorted_attachments != self.connection_attachments:
            object.__setattr__(self, "connection_attachments", sorted_attachments)
        if sorted_indexed != self.indexed_sources:
            object.__setattr__(self, "indexed_sources", sorted_indexed)
        if sorted_live != self.live_access_bindings:
            object.__setattr__(self, "live_access_bindings", sorted_live)

        return self
