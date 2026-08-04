# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Hybrid Ask evidence, citation and Ask Run V2 contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from local_workspace_application.workspaces.ask_models import AskError, AskRunStatus
from local_workspace_application.workspaces.knowledge_configuration_models import QueryPolicyModeV2

_EVIDENCE_ID_INDEXED_PREFIX = "idx:"
_EVIDENCE_ID_LIVE_PREFIX = "live:"


class EvidenceTypeV1(StrEnum):
    INDEXED = "indexed"
    LIVE = "live"


class AskAudienceV1(StrEnum):
    PERSONAL = "personal"
    SHARED = "shared"


class AskCitationLocationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int | None = None


class IndexedWorkspaceEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    evidence_type: Literal[EvidenceTypeV1.INDEXED] = EvidenceTypeV1.INDEXED
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    safe_display_name: str = Field(..., min_length=1)
    retrieved_at: datetime
    content: str = ""
    content_hash: str = Field(..., min_length=1)
    audience: AskAudienceV1
    source_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    chunk_id: str | None = None
    location: AskCitationLocationV1 | None = None
    score: float | None = None
    safe_source_label: str | None = None
    indexed_source_binding_id: str | None = None

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not value.startswith(_EVIDENCE_ID_INDEXED_PREFIX):
            raise ValueError("indexed_evidence_id_prefix_required")
        return value


class LiveWorkspaceEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    evidence_type: Literal[EvidenceTypeV1.LIVE] = EvidenceTypeV1.LIVE
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    safe_display_name: str = Field(..., min_length=1)
    retrieved_at: datetime
    content: str = ""
    content_hash: str = Field(..., min_length=1)
    audience: AskAudienceV1
    live_access_binding_id: str = Field(..., min_length=1)
    connection_ref: str = Field(..., min_length=1)
    capability_id: str = Field(..., min_length=1)
    remote_resource_id: str | None = None
    remote_item_id: str | None = None
    provider_id: str = Field(..., min_length=1)
    integration_kind: str = Field(..., min_length=1)
    call_id: str = Field(..., min_length=1)
    remote_updated_at: datetime | None = None
    safe_locator: str | None = None
    truncated: bool = False
    execution_receipt_id: str | None = None

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not value.startswith(_EVIDENCE_ID_LIVE_PREFIX):
            raise ValueError("live_evidence_id_prefix_required")
        return value


WorkspaceEvidenceV1 = Annotated[
    IndexedWorkspaceEvidenceV1 | LiveWorkspaceEvidenceV1,
    Field(discriminator="evidence_type"),
]


class PersistedIndexedEvidenceV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    evidence_type: Literal[EvidenceTypeV1.INDEXED] = EvidenceTypeV1.INDEXED
    safe_display_name: str = Field(..., min_length=1)
    retrieved_at: datetime
    content_hash: str = Field(..., min_length=1)
    audience: AskAudienceV1
    source_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    chunk_id: str | None = None
    location: AskCitationLocationV1 | None = None
    score: float | None = None
    safe_source_label: str | None = None
    indexed_source_binding_id: str | None = None

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not value.startswith(_EVIDENCE_ID_INDEXED_PREFIX):
            raise ValueError("indexed_evidence_id_prefix_required")
        return value


class PersistedLiveEvidenceProvenanceV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    evidence_type: Literal[EvidenceTypeV1.LIVE] = EvidenceTypeV1.LIVE
    safe_display_name: str = Field(..., min_length=1)
    retrieved_at: datetime
    content_hash: str = Field(..., min_length=1)
    audience: AskAudienceV1
    provider_id: str = Field(..., min_length=1)
    live_access_binding_id: str = Field(..., min_length=1)
    connection_ref: str = Field(..., min_length=1)
    capability_id: str = Field(..., min_length=1)
    remote_resource_id: str | None = None
    remote_item_id: str | None = None
    remote_updated_at: datetime | None = None
    truncated: bool = False
    call_id: str = Field(..., min_length=1)

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not value.startswith(_EVIDENCE_ID_LIVE_PREFIX):
            raise ValueError("live_evidence_id_prefix_required")
        return value


PersistedAskEvidenceV2 = Annotated[
    PersistedIndexedEvidenceV2 | PersistedLiveEvidenceProvenanceV2,
    Field(discriminator="evidence_type"),
]


class IndexedWorkspaceCitationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    evidence_type: Literal[EvidenceTypeV1.INDEXED] = EvidenceTypeV1.INDEXED
    safe_display_name: str = Field(..., min_length=1)
    excerpt: str = ""
    retrieved_at: datetime
    document_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)
    chunk_id: str | None = None
    score: float | None = None
    location: AskCitationLocationV1 | None = None

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not value.startswith(_EVIDENCE_ID_INDEXED_PREFIX):
            raise ValueError("indexed_citation_evidence_id_prefix_required")
        return value


class LiveWorkspaceCitationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    evidence_type: Literal[EvidenceTypeV1.LIVE] = EvidenceTypeV1.LIVE
    safe_display_name: str = Field(..., min_length=1)
    retrieved_at: datetime
    provider_id: str = Field(..., min_length=1)
    connection_safe_label: str = Field(..., min_length=1)
    capability_id: str = Field(..., min_length=1)
    remote_resource_id: str | None = None
    remote_item_id: str | None = None
    remote_updated_at: datetime | None = None
    call_id: str = Field(..., min_length=1)
    receipt_id: str | None = None

    @field_validator("evidence_id")
    @classmethod
    def _validate_evidence_id(cls, value: str) -> str:
        if not value.startswith(_EVIDENCE_ID_LIVE_PREFIX):
            raise ValueError("live_citation_evidence_id_prefix_required")
        return value


WorkspaceCitationV1 = Annotated[
    IndexedWorkspaceCitationV1 | LiveWorkspaceCitationV1,
    Field(discriminator="evidence_type"),
]


class LiveExecutionReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    call_id: str = Field(..., min_length=1)
    live_access_binding_id: str = Field(..., min_length=1)
    capability_id: str = Field(..., min_length=1)
    started_at: datetime
    completed_at: datetime
    item_count: int = Field(..., ge=0)
    byte_count: int = Field(..., ge=0)
    content_hash: str = Field(..., min_length=1)
    truncated: bool = False
    normalized_outcome: str = Field(..., min_length=1)


class HybridAskIndexedRetrievalStatusV1(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class HybridAskLiveExecutionStatusV1(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class HybridAskTruncationStateV1(StrEnum):
    NONE = "none"
    INDEXED = "indexed"
    LIVE = "live"
    BOTH = "both"


class WorkspaceAskRunV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_schema_version: Literal[2] = 2
    citation_schema_version: Literal[2] = 2
    run_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    status: AskRunStatus
    query_mode: QueryPolicyModeV2
    configuration_revision: int = Field(..., ge=0)
    plan_id: str = Field(..., min_length=1)
    answer: str | None = None
    citations: list[WorkspaceCitationV1] = Field(default_factory=list)
    persisted_evidence: list[PersistedAskEvidenceV2] = Field(default_factory=list)
    execution_receipts: list[LiveExecutionReceiptV1] = Field(default_factory=list)
    indexed_retrieval_status: HybridAskIndexedRetrievalStatusV1 = (
        HybridAskIndexedRetrievalStatusV1.SKIPPED
    )
    live_execution_status: HybridAskLiveExecutionStatusV1 = (
        HybridAskLiveExecutionStatusV1.SKIPPED
    )
    truncation_state: HybridAskTruncationStateV1 = HybridAskTruncationStateV1.NONE
    partial_failure: bool = False
    created_at: datetime
    completed_at: datetime | None = None
    error: AskError | None = None

    @model_validator(mode="after")
    def _validate_run_integrity(self) -> Self:
        evidence_by_id: dict[str, EvidenceTypeV1] = {}
        live_evidence_by_id: dict[str, PersistedLiveEvidenceProvenanceV2] = {}
        live_evidence_by_call_id: dict[str, PersistedLiveEvidenceProvenanceV2] = {}
        for item in self.persisted_evidence:
            if item.evidence_id in evidence_by_id:
                raise ValueError("duplicate_evidence_id")
            evidence_by_id[item.evidence_id] = item.evidence_type
            if isinstance(item, PersistedLiveEvidenceProvenanceV2):
                live_evidence_by_id[item.evidence_id] = item
                existing = live_evidence_by_call_id.get(item.call_id)
                if existing is not None and existing.evidence_id != item.evidence_id:
                    raise ValueError("duplicate_live_call_id")
                live_evidence_by_call_id[item.call_id] = item

        receipt_by_id: dict[str, LiveExecutionReceiptV1] = {}
        for receipt in self.execution_receipts:
            if receipt.receipt_id in receipt_by_id:
                raise ValueError("duplicate_receipt_id")
            if receipt.run_id != self.run_id:
                raise ValueError("receipt_run_id_mismatch")
            receipt_by_id[receipt.receipt_id] = receipt
            live_evidence = live_evidence_by_call_id.get(receipt.call_id)
            if live_evidence is None:
                raise ValueError("receipt_unknown_live_call")
            if receipt.live_access_binding_id != live_evidence.live_access_binding_id:
                raise ValueError("receipt_binding_mismatch")
            if receipt.capability_id != live_evidence.capability_id:
                raise ValueError("receipt_capability_mismatch")

        for citation in self.citations:
            evidence_type = evidence_by_id.get(citation.evidence_id)
            if evidence_type is None:
                raise ValueError("citation_evidence_not_found")
            if evidence_type is not citation.evidence_type:
                raise ValueError("citation_evidence_type_mismatch")
            if isinstance(citation, LiveWorkspaceCitationV1):
                live_evidence = live_evidence_by_id.get(citation.evidence_id)
                if live_evidence is None:
                    raise ValueError("citation_live_evidence_not_found")
                if citation.call_id != live_evidence.call_id:
                    raise ValueError("citation_live_call_id_mismatch")
                if citation.receipt_id is not None:
                    receipt = receipt_by_id.get(citation.receipt_id)
                    if receipt is None:
                        raise ValueError("citation_receipt_not_found")
                    if receipt.call_id != citation.call_id:
                        raise ValueError("citation_receipt_call_id_mismatch")

        if self.status is AskRunStatus.COMPLETED:
            if not self.citations:
                raise ValueError("completed_run_requires_citations")
            if self.query_mode is QueryPolicyModeV2.HYBRID:
                has_indexed = any(
                    citation.evidence_type is EvidenceTypeV1.INDEXED
                    for citation in self.citations
                )
                has_live = any(
                    citation.evidence_type is EvidenceTypeV1.LIVE
                    for citation in self.citations
                )
                if not has_indexed or not has_live:
                    raise ValueError("completed_hybrid_requires_indexed_and_live_citations")

        return self

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        for item in data.get("persisted_evidence", []):
            if isinstance(item, dict) and item.get("evidence_type") == EvidenceTypeV1.LIVE.value:
                _assert_no_forbidden_live_body_fields(item)
        for item in data.get("citations", []):
            if isinstance(item, dict) and item.get("evidence_type") == EvidenceTypeV1.LIVE.value:
                _assert_no_forbidden_live_body_fields(item)
        return data


_FORBIDDEN_DURABLE_LIVE_FIELDS = frozenset(
    {
        "content",
        "excerpt",
        "raw_provider_body",
        "structured_provider_result",
        "credentials",
        "tokens",
        "private_headers",
        "provider_client",
    }
)


def _assert_no_forbidden_live_body_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _FORBIDDEN_DURABLE_LIVE_FIELDS:
                raise ValueError("durable_live_body_field_forbidden")
            _assert_no_forbidden_live_body_fields(nested)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_live_body_fields(item)
