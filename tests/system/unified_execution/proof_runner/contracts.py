# © Artur Czarnecki. All rights reserved.

"""Typed contracts for UE-11G-C1 real agentic production certification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProofConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://local_workspace:8020"
    api_key: str = "ue-11g-c1-certification-secret"
    tenant_id: str = "tenant-ue-11g-c1"
    workspace_id: str = "ue-11g-c1-workspace"
    collection_id: str = "ue-11g-c1-collection"
    capability: str = "local.workspace.search"
    agent_id: str = "local_search"
    strategy: Literal["AGENTIC"] = "AGENTIC"
    llm_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:latest"
    fixture_root: str = "/cert-fixtures/workspace"
    request_timeout_seconds: float = 240.0
    readiness_timeout_seconds: float = 900.0
    ollama_base_url: str = "http://ollama:11434"
    otlp_log_path: str = "/var/lib/otelcol/lkw-otlp-logs.jsonl"
    runtime_events_db_path: str = "/lkw-data/data/sqlite/intergrax_runtime_events.db"


class AgentInvocationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    run_id: str
    total_llm_tokens: int = 0
    total_tool_calls: int = 0


class ApplicationRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    task_id: str
    terminal_status: str
    total_llm_tokens: int = 0
    agent_invocations: list[AgentInvocationSummary] = Field(default_factory=list)


class SearchSummaryDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_results: int | None = None
    evidence_count: int | None = None
    source_refs: list[str] | None = None
    reason: str | None = None


class LkwEvidenceSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    capability: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    diagnostics: dict[str, object] = Field(default_factory=dict)


class RuntimeToolEventEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str
    requested: int = 0
    completed: int = 0


class RuntimeEventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    tool_events_total: int = 0
    tools: list[RuntimeToolEventEntry] = Field(default_factory=list)


class LkwRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: str
    state: str
    answer: str | None = None
    agent_id: str | None = None
    application_run_summary: ApplicationRunSummary | None = None
    lkw_evidence: LkwEvidenceSlice | None = None
    runtime_event_summary: RuntimeEventSummary | None = None


class OllamaModelEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    digest_present: bool
    listed_after_run: bool


class OtlpIdentityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str | None = None
    execution_id: str | None = None
    attempt_id: str | None = None
    capability: str | None = None
    agent_id: str | None = None
    tool_id: str | None = None
    event_count: int = 0


class CertificationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    http_status: int
    task_id: str
    run_id: str
    attempt_id: str | None = None
    execution_id: str | None = None
    strategy: Literal["AGENTIC"] = "AGENTIC"
    capability: str
    agent_id: str
    llm_provider: str
    embedding_model: str
    application: str = "local_workspace_application"
    endpoint: str = "/v1/local_workspace/run"
    runtime_event_persistence: str = "sqlite_runtime_events"
    ollama: OllamaModelEvidence
    otlp: OtlpIdentityEvidence
    budget_tokens: int = 0
    authority_evidence: str = "sqlite_runtime_events_attempt_id"
    functional_oracle_pass: bool = False
    functional_expected: str | None = None
    functional_actual_bounded: str | None = None


class DiagnosticCheckResultProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: str
    factual_claim: str


class FunctionalDiagnosticSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_status: Literal["PASS", "FAIL", "BLOCKED"]
    persistence_backend: str
    durable: bool
    evidence_kinds: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    validation_id: str | None = None
    functional_expected: str
    functional_actual_bounded: str
    diagnostic_specification_id: str | None = None
    diagnostic_specification_version: int | None = None
    diagnostic_first_proven_failure: str | None = None
    diagnostic_check_results: list[DiagnosticCheckResultProjection] = Field(default_factory=list)
    diagnostic_supporting_evidence_refs: list[str] = Field(default_factory=list)
    diagnostic_limitations: list[str] = Field(default_factory=list)
    failure_stage: str | None = None
    confidence: Literal["PROVEN", "INSUFFICIENT"]
    blocked_reason: str | None = None


class ProofReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proof_id: Literal["UE-11G-C1"] = "UE-11G-C1"
    verdict: Literal["PASS", "FAIL", "PARTIAL", "BLOCKED"]
    evidence: CertificationEvidence | None = None
    failure_reason: str | None = None
    functional_diagnostic: FunctionalDiagnosticSection | None = None
    r4_result: Literal["PASS", "PARTIAL", "FAIL", "BLOCKED"] | None = None
