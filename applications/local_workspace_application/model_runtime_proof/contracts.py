# © Artur Czarnecki. All rights reserved.

"""Typed contracts for LKW model runtime portability proof."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROOF_SCHEMA_VERSION = "lkw.model_runtime_portability.proof.v2"
PROOF_TASK_ID = "LKW-MODEL-RUNTIME-1"
PROOF_CLASSIFICATION = "controlled local-provider live LKW product proof for exact Ollama and vLLM provider/model pairs"
FIXTURE_MARKER = "MODEL-RUNTIME-8421"
FIXTURE_TEXT = (
    f"The LKW model runtime portability verification code is {FIXTURE_MARKER}."
)
BASIC_GENERATION_MARKER = "LKW-RUNTIME-BASIC-7319"
PLANNING_MESSAGE = "dodaj https://example.com/runtime-proof do workspace magazyn"
PLANNING_URL = "https://example.com/runtime-proof"
ASK_QUESTION = "What is the LKW model runtime portability verification code?"
WORKSPACE_SEARCH_TOOL = "local.workspace.search"
MANAGED_WORKSPACE_USER_ID = "lkw.managed_workspace"


class ProofFailureCode(StrEnum):
    PROVIDER_UNREACHABLE = "provider_unreachable"
    PROVIDER_MODEL_MISSING = "provider_model_missing"
    PROVIDER_MODEL_MISMATCH = "provider_model_mismatch"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    BASIC_GENERATION_FAILED = "basic_generation_failed"
    STRUCTURED_PLANNING_UNSUPPORTED = "structured_planning_unsupported"
    STRUCTURED_PLANNING_FAILED = "structured_planning_failed"
    TOOL_CALL_UNSUPPORTED = "tool_call_unsupported"
    TOOL_CALL_MISSING = "tool_call_missing"
    TOOL_CALL_MULTIPLE = "tool_call_multiple"
    TOOL_CALL_INVALID = "tool_call_invalid"
    TOOL_CALL_UNEXPECTED_TOOL = "tool_call_unexpected_tool"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    GROUNDED_ASK_FAILED = "grounded_ask_failed"
    CITATION_MISSING = "citation_missing"
    CITATION_MARKER_MISSING = "citation_marker_missing"
    EMBEDDING_IDENTITY_CHANGED = "embedding_identity_changed"
    VECTOR_COLLECTION_CHANGED = "vector_collection_changed"
    VECTOR_COUNT_CHANGED = "vector_count_changed"
    UNEXPECTED_REINDEX = "unexpected_reindex"
    PROOF_SECRET_LEAK_DETECTED = "proof_secret_leak_detected"
    CONFIG_INVALID = "config_invalid"
    INDEX_FIXTURE_FAILED = "index_fixture_failed"


class StageStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    NOT_RUN = "NOT_RUN"


class ProofOverallStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FixtureRecord(_Frozen):
    marker: str = FIXTURE_MARKER
    fixture_text_excerpt: str = FIXTURE_TEXT[:80]
    tenant_id: str | None = None
    workspace_id: str | None = None
    input_id: str | None = None
    source_id: str | None = None
    operation_id: str | None = None
    document_id: str | None = None
    content_hash: str | None = None
    indexing_operations: int = 0
    chunk_count: int | None = None
    index_path_classification: str = "managed_file_knowledge_intake"


class EmbeddingIdentityRecord(_Frozen):
    provider: str
    model: str
    dimensions: int | None = None


class IndexIdentityRecord(_Frozen):
    tenant_id: str
    workspace_id: str
    source_id: str | None = None
    document_id: str | None = None
    content_hash: str | None = None
    collection_identity: str
    vector_count: int
    chunk_count: int | None = None
    embedding: EmbeddingIdentityRecord


class ProviderStageSummary(_Frozen):
    health: StageStatus = StageStatus.NOT_RUN
    generation: StageStatus = StageStatus.NOT_RUN
    structured_plan: StageStatus = StageStatus.NOT_RUN
    tool_call: StageStatus = StageStatus.NOT_RUN
    tool_execution: StageStatus = StageStatus.NOT_RUN
    grounded_ask: StageStatus = StageStatus.NOT_RUN
    citation: StageStatus = StageStatus.NOT_RUN


class ProviderQualificationResult(_Frozen):
    provider: str
    configured_model: str
    resolved_model: str | None = None
    server_model: str | None = None
    server_model_digest: str | None = None
    adapter_class: str | None = None
    server_version: str | None = None
    base_url_classification: str | None = None
    health_status: StageStatus = StageStatus.NOT_RUN
    basic_generation_status: StageStatus = StageStatus.NOT_RUN
    structured_planning_status: StageStatus = StageStatus.NOT_RUN
    tool_call_status: StageStatus = StageStatus.NOT_RUN
    tool_execution_status: StageStatus = StageStatus.NOT_RUN
    grounded_ask_status: StageStatus = StageStatus.NOT_RUN
    citation_status: StageStatus = StageStatus.NOT_RUN
    tool_choice_mode: Literal["forced", "automatic"] | None = None
    basic_generation_excerpt: str | None = None
    planning_validation: str | None = None
    ask_answer_excerpt: str | None = None
    citation_source_id: str | None = None
    citation_excerpt: str | None = None
    ask_run_persisted: bool = False
    http_ask_status_code: int | None = None
    resolved_through_canonical_resolver: bool = False
    session_adapter_object_id: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    failure_code: ProofFailureCode | None = None
    safe_error_type: str | None = None
    safe_error_excerpt: str | None = None

    @property
    def stages(self) -> ProviderStageSummary:
        return ProviderStageSummary(
            health=self.health_status,
            generation=self.basic_generation_status,
            structured_plan=self.structured_planning_status,
            tool_call=self.tool_call_status,
            tool_execution=self.tool_execution_status,
            grounded_ask=self.grounded_ask_status,
            citation=self.citation_status,
        )


class IndexInvarianceResult(_Frozen):
    embedding_identity: StageStatus = StageStatus.NOT_RUN
    collection_identity: StageStatus = StageStatus.NOT_RUN
    vector_count: StageStatus = StageStatus.NOT_RUN
    source_identity: StageStatus = StageStatus.NOT_RUN
    document_identity: StageStatus = StageStatus.NOT_RUN
    content_hash: StageStatus = StageStatus.NOT_RUN
    chunk_count: StageStatus = StageStatus.NOT_RUN
    no_reindex: StageStatus = StageStatus.NOT_RUN


class RepositoryStateRecord(_Frozen):
    repository_head_at_proof: str | None = None
    repository_head_role: str = "pre_evidence_commit_head"
    working_tree_classification: str = "unavailable"
    task_owned_dirty_paths: tuple[str, ...] = ()
    unrelated_dirty_paths: tuple[str, ...] = ()


class ModelRuntimeProofResult(_Frozen):
    schema_version: str = PROOF_SCHEMA_VERSION
    proof_id: str
    task_id: str = PROOF_TASK_ID
    started_at: datetime
    completed_at: datetime | None = None
    repository_commit: str | None = None
    repository_state: RepositoryStateRecord = Field(
        default_factory=RepositoryStateRecord
    )
    proof_classification: str = PROOF_CLASSIFICATION
    vllm_provisioning_classification: str | None = None
    fixture: FixtureRecord = Field(default_factory=FixtureRecord)
    index_identity: IndexIdentityRecord | None = None
    embedding_identity_before: EmbeddingIdentityRecord | None = None
    embedding_identity_after: EmbeddingIdentityRecord | None = None
    provider_results: dict[str, ProviderQualificationResult] = Field(
        default_factory=dict
    )
    index_invariance: IndexInvarianceResult = Field(
        default_factory=IndexInvarianceResult
    )
    overall_status: ProofOverallStatus = ProofOverallStatus.FAIL
    limitations: tuple[str, ...] = ()
    extra_safe: dict[str, Any] = Field(default_factory=dict)
