# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AdasOperationalEnvelope,
    AgentCandidateDraft,
    AgentCandidateEvaluationResult,
    AgentCandidateEvidenceBundle,
    AgentCandidateRecord,
    AgentCandidateStatus,
    AgentDesignObjective,
    AgentDesignSearchRun,
    AggregateResourceLimits,
    BaselineConfigurationRef,
    ConfigurationDelta,
    ResourceAccountingEvidence,
    SearchSelectionOutcome,
    canonical_record_json,
    compute_evidence_bundle_digest,
    verify_evidence_bundle_digest,
)
from intergrax.runtime.adaptive.contracts import ProfileArtifactType
from pydantic import ValidationError

_DIGEST = "a" * 64
_NOW = datetime.now(UTC)


def sample_envelope(**overrides: Any) -> AdasOperationalEnvelope:
    payload = {
        "created_at": _NOW,
        "created_by": "operator",
        "tenant_id": "tenant-a",
        "workspace_id": "ws-a",
        "accepted_scope_ref": "scope/lab",
        "retention_policy": "lab-default",
        **overrides,
    }
    return AdasOperationalEnvelope.model_validate(payload)


def sample_baseline() -> BaselineConfigurationRef:
    return BaselineConfigurationRef(
        baseline_version_id="baseline-v1",
        baseline_configuration_digest=_DIGEST,
        artifact_type=ProfileArtifactType.ORCHESTRATION,
    )


def sample_limits() -> AggregateResourceLimits:
    return AggregateResourceLimits(
        aggregate_token_allowance=1000.0,
        aggregate_monetary_allowance=5.0,
        deadline_at=_NOW + timedelta(hours=1),
    )


def sample_objective() -> AgentDesignObjective:
    return AgentDesignObjective(
        objective_id="obj-1",
        tenant_id="tenant-a",
        workspace_id="ws-a",
        task_class="merge-lab",
        target_capability="orchestration.profile",
        envelope=sample_envelope(),
        baseline=sample_baseline(),
        goal_summary="Improve merge strategy",
        success_metric="utility",
        dataset_id="ds-1",
        dataset_content_digest=_DIGEST,
        acceptance_criteria_ref="crit-1",
        aggregate_limits=sample_limits(),
        created_by="operator",
        created_at=_NOW,
    )


def sample_delta() -> ConfigurationDelta:
    return ConfigurationDelta(prior_value=MergeStrategy.CONCAT, replacement_value=MergeStrategy.LAST_WINS)


def sample_record(
    *,
    sequence: int = 0,
    status: AgentCandidateStatus = AgentCandidateStatus.DRAFT,
    candidate_id: str = "cand-1",
    snapshot_id: str | None = None,
) -> AgentCandidateRecord:
    return AgentCandidateRecord(
        snapshot_id=snapshot_id or f"snap-{candidate_id}-{sequence}",
        sequence=sequence,
        candidate_id=candidate_id,
        objective_id="obj-1",
        search_run_id="run-1",
        tenant_id="tenant-a",
        workspace_id="ws-a",
        lineage_id="line-1",
        status=status,
        envelope=sample_envelope(),
        baseline=sample_baseline(),
        candidate_configuration_digest="b" * 64,
        configuration_delta=sample_delta(),
        created_by="operator",
        created_at=_NOW,
    )


def test_candidate_status_canonical_values() -> None:
    assert AgentCandidateStatus.DRAFT.value == "draft"
    assert AgentCandidateStatus.PROMOTED.value == "promoted"
    assert SearchSelectionOutcome.RETAIN_BASELINE.value == "retain_baseline"


def test_w1_objective_construction() -> None:
    objective = sample_objective()
    assert objective.aggregate_limits.max_candidates == 1


def test_invalid_scope_rejection() -> None:
    envelope = sample_envelope(tenant_id="tenant-b")
    with pytest.raises(ValidationError):
        AgentDesignObjective(
            objective_id="obj-1",
            tenant_id="tenant-a",
            workspace_id="ws-a",
            task_class="merge-lab",
            target_capability="cap",
            envelope=envelope,
            baseline=sample_baseline(),
            goal_summary="g",
            success_metric="m",
            dataset_id="ds",
            dataset_content_digest=_DIGEST,
            acceptance_criteria_ref="c",
            aggregate_limits=sample_limits(),
            created_by="op",
            created_at=_NOW,
        )


def test_invalid_digest_and_no_op_delta() -> None:
    with pytest.raises(ValidationError):
        ConfigurationDelta(prior_value=MergeStrategy.CONCAT, replacement_value=MergeStrategy.CONCAT)
    with pytest.raises(ValidationError):
        BaselineConfigurationRef(
            baseline_version_id="v",
            baseline_configuration_digest="BAD",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
        )


def test_deep_immutability_and_canonical_json() -> None:
    record = sample_record().model_copy(update={"agent_contract_snapshot": {"k": "v"}})
    dumped = canonical_record_json(record)
    assert dumped["status"] == "draft"
    record.agent_contract_snapshot["k"] = "mutated"
    latest = sample_record()
    assert latest.agent_contract_snapshot == {}


def test_evaluation_unknown_metrics_and_passed_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        AgentCandidateEvaluationResult(candidate_id="cand-1", passed=True)
    result = AgentCandidateEvaluationResult(
        candidate_id="cand-1",
        passed=True,
        quality_score=0.9,
        cost_normalized=0.1,
        latency_ms=10,
        total_tokens=100,
        utility_score=0.8,
        golden_pass_rate=1.0,
        negative_pass_rate=1.0,
        adversarial_pass_rate=1.0,
        execution_ref="exec-1",
        evaluation_ref="eval-1",
        evidence_complete=True,
    )
    assert result.tool_calls is None


def test_evidence_digest_stability_and_rejected_without_eval() -> None:
    record = sample_record(status=AgentCandidateStatus.REJECTED)
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-1",
        candidate_id=record.candidate_id,
        objective_id=record.objective_id,
        tenant_id=record.tenant_id,
        workspace_id=record.workspace_id,
        created_at=_NOW,
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest="b" * 64,
        candidate_record=record,
        static_gate_result={"passed": False, "reason": "rejected"},
        assembled_at=_NOW,
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    verify_evidence_bundle_digest(sealed)
    with pytest.raises(ValueError):
        verify_evidence_bundle_digest(sealed.model_copy(update={"bundle_digest": "c" * 64}))
    with pytest.raises(ValidationError):
        AgentCandidateEvidenceBundle(
            bundle_id="bundle-2",
            candidate_id=record.candidate_id,
            objective_id=record.objective_id,
            tenant_id=record.tenant_id,
            workspace_id=record.workspace_id,
            created_at=_NOW,
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest="b" * 64,
            candidate_record=record,
            evaluation_result=AgentCandidateEvaluationResult(candidate_id=record.candidate_id, passed=False),
            static_gate_result={"passed": False},
            assembled_at=_NOW,
        )


def test_draft_w1_absent_scaffold_fields() -> None:
    draft = AgentCandidateDraft(
        candidate_id="cand-1",
        objective_id="obj-1",
        search_run_id="run-1",
        tenant_id="tenant-a",
        workspace_id="ws-a",
        lineage_id="line-1",
        generated_by="lab",
        generation_strategy="rule",
        design_rationale="test",
        baseline=sample_baseline(),
        candidate_configuration_digest="b" * 64,
        configuration_delta=sample_delta(),
        created_at=_NOW,
    )
    assert draft.scaffold_pattern is None
    assert draft.scaffold_output_ref is None


def test_search_run_terminal_rules() -> None:
    with pytest.raises(ValidationError):
        AgentDesignSearchRun(
            search_run_id="run-1",
            objective_id="obj-1",
            tenant_id="tenant-a",
            workspace_id="ws-a",
            task_class="t",
            envelope=sample_envelope(),
            strategy_name="rule",
            status="completed",
            baseline=sample_baseline(),
            resource_accounting=ResourceAccountingEvidence(),
            started_at=_NOW,
        )
