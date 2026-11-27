# © Artur Czarnecki. All rights reserved.

"""Canonical ADAS typed contracts (AHI-ADAS-10, W1 configuration-only LAB profile)."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.canonical_payload_hash import canonical_json_bytes
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.adaptive.contracts import ProfileArtifactType

ADAS_CONTRACT_SCHEMA_VERSION = "1.0.0"
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")
_W1_MERGE_STRATEGIES = frozenset({MergeStrategy.LAST_WINS, MergeStrategy.CONCAT})


def _non_blank(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    return normalized


def _validate_sha256_digest(value: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_HEX.fullmatch(normalized):
        raise ValueError("digest must be lowercase 64-character hexadecimal SHA-256")
    return normalized


def _finite_non_negative(value: float, *, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _finite_non_negative_int(value: int, *, field: str) -> int:
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class AgentCandidateStatus(str, Enum):  # noqa: UP042 — canonical ADAS serialized enum values
    DRAFT = "draft"
    SCAFFOLDED = "scaffolded"
    STATIC_GATE_FAILED = "static_gate_failed"
    STATIC_GATE_PASSED = "static_gate_passed"
    EVALUATED = "evaluated"
    REJECTED = "rejected"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    RETIRED = "retired"
    ARCHIVED = "archived"


class SearchSelectionOutcome(str, Enum):  # noqa: UP042
    SELECT_CANDIDATE = "select_candidate"
    RETAIN_BASELINE = "retain_baseline"
    ABSTAIN = "abstain"


class AdasOperationalEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ADAS_CONTRACT_SCHEMA_VERSION
    created_at: datetime
    updated_at: datetime | None = None
    created_by: str
    tenant_id: str
    workspace_id: str
    accepted_scope_ref: str
    lab_only_posture: bool = True
    approved_by: str | None = None
    approval_id: str | None = None
    policy_context_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None
    artifact_hash: str | None = None
    source_hash: str | None = None
    evaluation_bundle_id: str | None = None
    risk_class: str | None = None
    retention_policy: str

    @field_validator("created_by", "tenant_id", "workspace_id", "accepted_scope_ref", "retention_policy")
    @classmethod
    def _required_identities(cls, value: str) -> str:
        return _non_blank(value, field="identity")

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_datetime(value)

    @field_validator("artifact_hash", "source_hash")
    @classmethod
    def _optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256_digest(value)

    @model_validator(mode="after")
    def _lab_posture(self) -> Self:
        if not self.lab_only_posture:
            raise ValueError("W1 requires LAB-only posture")
        return self


class AggregateResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_candidates: int = 1
    max_search_iterations: int = 1
    max_root_executions: int = 2
    aggregate_token_allowance: float
    aggregate_monetary_allowance: float
    deadline_at: datetime

    @field_validator("max_candidates", "max_search_iterations", "max_root_executions")
    @classmethod
    def _w1_limits(cls, value: int) -> int:
        return _finite_non_negative_int(value, field="limit")

    @field_validator("aggregate_token_allowance", "aggregate_monetary_allowance")
    @classmethod
    def _budgets(cls, value: float) -> float:
        return _finite_non_negative(value, field="budget")

    @field_validator("deadline_at")
    @classmethod
    def _deadline(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _enforce_w1_caps(self) -> Self:
        if self.max_candidates != 1 or self.max_search_iterations != 1 or self.max_root_executions != 2:
            raise ValueError("W1 aggregate limits must cap candidates, iterations and root executions")
        return self


class BaselineConfigurationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_version_id: str
    baseline_configuration_digest: str
    artifact_type: ProfileArtifactType

    @field_validator("baseline_version_id")
    @classmethod
    def _version(cls, value: str) -> str:
        return _non_blank(value, field="baseline_version_id")

    @field_validator("baseline_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)


class ConfigurationDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted_field: str = "merge_strategy"
    prior_value: MergeStrategy
    replacement_value: MergeStrategy

    @model_validator(mode="after")
    def _non_noop(self) -> Self:
        if self.prior_value == self.replacement_value:
            raise ValueError("configuration delta must not be a no-op")
        if self.prior_value not in _W1_MERGE_STRATEGIES or self.replacement_value not in _W1_MERGE_STRATEGIES:
            raise ValueError("W1 admits only last_wins and concat merge strategies")
        return self


class ResourceAccountingEvidence(BaseModel):
    """Explicit accounting; omit fields that were not measured (no default zero)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens_consumed: float | None = None
    monetary_consumed: float | None = None
    root_executions_consumed: int | None = None

    @field_validator("tokens_consumed", "monetary_consumed")
    @classmethod
    def _measured_floats(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(value, field="consumption")

    @field_validator("root_executions_consumed")
    @classmethod
    def _measured_int(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _finite_non_negative_int(value, field="root_executions_consumed")


class AgentDesignObjective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: str
    tenant_id: str
    workspace_id: str
    application_id: str | None = None
    task_class: str
    target_capability: str
    envelope: AdasOperationalEnvelope
    baseline: BaselineConfigurationRef
    goal_summary: str
    success_metric: str
    dataset_id: str
    dataset_content_digest: str
    acceptance_criteria_ref: str
    admitted_configuration_surface: tuple[str, ...] = ("merge_strategy",)
    allowed_merge_strategies: tuple[MergeStrategy, ...] = (
        MergeStrategy.LAST_WINS,
        MergeStrategy.CONCAT,
    )
    aggregate_limits: AggregateResourceLimits
    external_effects_prohibited: bool = True
    require_human_approval: bool = True
    created_by: str
    created_at: datetime

    @field_validator(
        "objective_id",
        "tenant_id",
        "workspace_id",
        "task_class",
        "target_capability",
        "goal_summary",
        "success_metric",
        "dataset_id",
        "acceptance_criteria_ref",
        "created_by",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("dataset_content_digest")
    @classmethod
    def _dataset_digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("created_at")
    @classmethod
    def _created(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _scope_and_w1(self) -> Self:
        if self.envelope.tenant_id != self.tenant_id:
            raise ValueError("envelope tenant_id must match objective tenant_id")
        if self.envelope.workspace_id != self.workspace_id:
            raise ValueError("envelope workspace_id must match objective workspace_id")
        if not self.external_effects_prohibited:
            raise ValueError("W1 objectives must prohibit external effects")
        if set(self.allowed_merge_strategies) != _W1_MERGE_STRATEGIES:
            raise ValueError("W1 allows only last_wins and concat merge strategies")
        return self


class AgentDesignSearchRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_run_id: str
    objective_id: str
    tenant_id: str
    workspace_id: str
    task_class: str
    envelope: AdasOperationalEnvelope
    strategy_name: str
    status: str
    baseline: BaselineConfigurationRef
    resource_accounting: ResourceAccountingEvidence
    selection_outcome: SearchSelectionOutcome | None = None
    started_at: datetime
    completed_at: datetime | None = None
    generated_candidate_ids: tuple[str, ...] = ()
    stop_reason: str | None = None

    @field_validator(
        "search_run_id",
        "objective_id",
        "tenant_id",
        "workspace_id",
        "task_class",
        "strategy_name",
        "status",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("started_at", "completed_at")
    @classmethod
    def _times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _scope_and_terminal(self) -> Self:
        if self.envelope.tenant_id != self.tenant_id or self.envelope.workspace_id != self.workspace_id:
            raise ValueError("envelope scope must match search run scope")
        if self.status in {"completed", "failed", "cancelled"}:
            if self.completed_at is None:
                raise ValueError("terminal search runs require completed_at")
            if self.stop_reason is None or not self.stop_reason.strip():
                raise ValueError("terminal search runs require stop_reason")
            if self.status == "completed" and self.selection_outcome is None:
                raise ValueError("completed search runs require selection_outcome")
            failed_or_cancelled = self.status in {"failed", "cancelled"}
            if failed_or_cancelled and self.selection_outcome == SearchSelectionOutcome.SELECT_CANDIDATE:
                raise ValueError("failed or cancelled runs must not claim candidate selection")
        return self


class AgentCandidateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    objective_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    parent_candidate_id: str | None = None
    lineage_id: str
    generated_by: str
    generation_strategy: str
    design_rationale: str
    baseline: BaselineConfigurationRef
    candidate_configuration_digest: str
    configuration_delta: ConfigurationDelta
    scaffold_pattern: None = None
    capabilities: tuple[str, ...] = ()
    contract_draft: dict[str, Any] = Field(default_factory=dict)
    prompt_assets: dict[str, Any] = Field(default_factory=dict)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    state_schema: None = None
    source_artifact_ref: str | None = None
    scaffold_output_ref: None = None
    created_at: datetime

    @field_validator(
        "candidate_id",
        "objective_id",
        "search_run_id",
        "tenant_id",
        "workspace_id",
        "lineage_id",
        "generated_by",
        "generation_strategy",
        "design_rationale",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("candidate_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("created_at")
    @classmethod
    def _created(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _forbid_executable_payloads(self) -> Self:
        forbidden = ("source", "eval", "exec", "__import__")
        for key in self.contract_draft:
            lowered = key.lower()
            if any(marker in lowered for marker in forbidden):
                raise ValueError("contract_draft must not carry executable payloads")
        return self


class AgentCandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str
    sequence: int
    candidate_id: str
    objective_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    parent_candidate_id: str | None = None
    lineage_id: str
    status: AgentCandidateStatus
    envelope: AdasOperationalEnvelope
    baseline: BaselineConfigurationRef
    candidate_configuration_digest: str
    configuration_delta: ConfigurationDelta
    agent_contract_snapshot: dict[str, Any] = Field(default_factory=dict)
    scaffold_pattern: None = None
    source_artifact_ref: str | None = None
    scaffold_output_ref: None = None
    eval_suite_id: str | None = None
    static_gate_result: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] | None = None
    utility_score: float | None = None
    baseline_utility_score: float | None = None
    utility_delta: float | None = None
    novelty_score: float | None = None
    complexity_score: float | None = None
    generation_cost: float | None = None
    evaluation_cost: float | None = None
    rejection_reason: str | None = None
    promotion_request_id: None = None
    created_by: str
    created_at: datetime
    archived_at: datetime | None = None

    @field_validator(
        "snapshot_id",
        "candidate_id",
        "objective_id",
        "search_run_id",
        "tenant_id",
        "workspace_id",
        "lineage_id",
        "created_by",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("candidate_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("sequence")
    @classmethod
    def _sequence(cls, value: int) -> int:
        return _finite_non_negative_int(value, field="sequence")

    @field_validator("created_at", "archived_at")
    @classmethod
    def _times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_datetime(value)

    @field_validator(
        "utility_score",
        "baseline_utility_score",
        "utility_delta",
        "novelty_score",
        "complexity_score",
        "generation_cost",
        "evaluation_cost",
    )
    @classmethod
    def _optional_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("numeric field must be finite")
        return value


class AgentCandidateEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    baseline_agent_id: str | None = None
    passed: bool
    quality_score: float | None = None
    cost_normalized: float | None = None
    latency_ms: int | None = None
    total_tokens: int | None = None
    step_count: int | None = None
    tool_calls: int | None = None
    llm_calls: int | None = None
    hitl_interventions: int | None = None
    regression_flags: tuple[str, ...] = ()
    utility_score: float | None = None
    baseline_utility_score: float | None = None
    utility_delta: float | None = None
    golden_pass_rate: float | None = None
    negative_pass_rate: float | None = None
    adversarial_pass_rate: float | None = None
    failure_reasons: tuple[str, ...] = ()
    execution_ref: str | None = None
    evaluation_ref: str | None = None
    evidence_complete: bool = False

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return _non_blank(value, field="candidate_id")

    @field_validator(
        "quality_score",
        "cost_normalized",
        "utility_score",
        "baseline_utility_score",
        "utility_delta",
        "golden_pass_rate",
        "negative_pass_rate",
        "adversarial_pass_rate",
    )
    @classmethod
    def _optional_metrics(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("metric must be finite")
        return value

    @field_validator("latency_ms", "total_tokens", "step_count", "tool_calls", "llm_calls", "hitl_interventions")
    @classmethod
    def _optional_int_metrics(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _finite_non_negative_int(value, field="metric")

    @model_validator(mode="after")
    def _w1_verdict_evidence(self) -> Self:
        if self.passed:
            required = (
                self.quality_score,
                self.cost_normalized,
                self.latency_ms,
                self.total_tokens,
                self.utility_score,
                self.golden_pass_rate,
                self.negative_pass_rate,
                self.adversarial_pass_rate,
                self.execution_ref,
                self.evaluation_ref,
            )
            if any(item is None for item in required):
                raise ValueError("successful evaluation verdict requires mandatory W1 evidence")
            if not self.evidence_complete:
                raise ValueError("passed evaluation requires evidence_complete")
        return self


class AgentCandidateEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    candidate_id: str
    objective_id: str
    tenant_id: str
    workspace_id: str
    search_run_id: str | None = None
    schema_version: str = ADAS_CONTRACT_SCHEMA_VERSION
    created_at: datetime
    created_by: str
    correlation_id: str | None = None
    trace_id: str | None = None
    evaluation_bundle_id: str | None = None
    risk_class: str | None = None
    retention_policy: str | None = None
    bundle_digest: str | None = None
    baseline_configuration_digest: str
    candidate_configuration_digest: str
    selection_outcome: SearchSelectionOutcome | None = None
    candidate_record: AgentCandidateRecord
    baseline_evaluation_result: AgentCandidateEvaluationResult | None = None
    evaluation_result: AgentCandidateEvaluationResult | None = None
    static_gate_result: dict[str, Any] | None = None
    policy_gate_result: dict[str, Any] | None = None
    cost_report: dict[str, Any] | None = None
    security_adversarial_result: dict[str, Any] | None = None
    approval_record: dict[str, Any] | None = None
    promotion_decision: None = None
    rollback_pointer: str | None = None
    assembled_at: datetime
    sealed: bool = False

    @field_validator("bundle_id", "candidate_id", "objective_id", "tenant_id", "workspace_id", "created_by")
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("baseline_configuration_digest", "candidate_configuration_digest", "bundle_digest")
    @classmethod
    def _digests(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256_digest(value)

    @field_validator("created_at", "assembled_at")
    @classmethod
    def _times(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _scope_and_rejection_shape(self) -> Self:
        record = self.candidate_record
        if record.tenant_id != self.tenant_id or record.workspace_id != self.workspace_id:
            raise ValueError("bundle scope must match nested candidate record scope")
        if record.candidate_id != self.candidate_id or record.objective_id != self.objective_id:
            raise ValueError("bundle lineage identities must match candidate record")
        if record.status == AgentCandidateStatus.REJECTED and self.evaluation_result is not None:
            raise ValueError("rejected candidates must not fabricate evaluation results")
        if self.bundle_digest is not None:
            verify_evidence_bundle_digest(self)
        return self


def canonical_record_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def compute_evidence_bundle_digest(bundle: AgentCandidateEvidenceBundle) -> str:
    payload = bundle.model_dump(mode="json")
    payload.pop("bundle_digest", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_evidence_bundle_digest(bundle: AgentCandidateEvidenceBundle) -> None:
    declared = bundle.bundle_digest
    if declared is None:
        raise ValueError("bundle_digest is required for verification")
    recomputed = compute_evidence_bundle_digest(bundle)
    if declared != recomputed:
        raise ValueError("bundle_digest mismatch")
