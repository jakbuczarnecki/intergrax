# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    ADAS_CONTRACT_SCHEMA_VERSION,
    AdasEvalComparisonReference,
    AdasEvalRunReference,
    AdasExecutionReference,
    AdasJsonValue,
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
    W1TerminalEvidenceOutcome,
    _FrozenAdasJsonObject,
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
        "correlation_id": "corr-1",
        "trace_id": "trace-1",
        "policy_context_id": "policy-1",
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


def sample_evaluation_mapping(
    *,
    passed: bool = False,
) -> dict[str, AdasJsonValue]:
    if not passed:
        return {
            "passed": False,
            "execution_ref": "exec-1",
            "evaluation_ref": "eval-1",
            "failure_reasons": ["policy"],
        }
    return {
        "passed": True,
        "quality_score": 0.9,
        "cost_normalized": 0.1,
        "latency_ms": 10,
        "total_tokens": 100,
        "utility_score": 0.8,
        "golden_pass_rate": 1.0,
        "negative_pass_rate": 1.0,
        "adversarial_pass_rate": 1.0,
        "execution_ref": "exec-1",
        "evaluation_ref": "eval-1",
        "evidence_complete": True,
        "baseline_execution_ref": {"execution_id": "base-exec-1"},
        "eval_run_ref": {"eval_run_id": "run-eval-1"},
        "eval_comparison_ref": {"comparison_id": "cmp-1"},
    }


def sample_record(
    *,
    sequence: int = 0,
    status: AgentCandidateStatus = AgentCandidateStatus.DRAFT,
    candidate_id: str = "cand-1",
    snapshot_id: str | None = None,
    static_gate_result: dict[str, AdasJsonValue] | None = None,
    evaluation_result: dict[str, AdasJsonValue] | None = None,
    rejection_reason: str | None = None,
    archived_at: datetime | None = None,
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
        static_gate_result=static_gate_result,
        evaluation_result=evaluation_result,
        rejection_reason=rejection_reason,
        archived_at=archived_at,
    )


def test_candidate_status_canonical_values() -> None:
    for member in AgentCandidateStatus:
        assert isinstance(member.value, str)
        assert member.value == member.value.lower()


def test_schema_version_and_objective_canonical_fields() -> None:
    objective = sample_objective()
    assert ADAS_CONTRACT_SCHEMA_VERSION == "1.0"
    assert objective.max_candidates == objective.aggregate_limits.max_candidates == 1
    assert objective.max_iterations == objective.aggregate_limits.max_search_iterations == 1
    assert objective.max_generation_cost is None
    assert objective.baseline_agent_id is None


def test_invalid_scope_and_aggregate_limit_contradiction() -> None:
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
    with pytest.raises(ValidationError):
        AgentDesignObjective(
            objective_id="obj-1",
            tenant_id="tenant-a",
            workspace_id="ws-a",
            task_class="merge-lab",
            target_capability="cap",
            envelope=sample_envelope(),
            baseline=sample_baseline(),
            goal_summary="g",
            success_metric="m",
            dataset_id="ds",
            dataset_content_digest=_DIGEST,
            acceptance_criteria_ref="c",
            aggregate_limits=sample_limits(),
            max_iterations=5,
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
    with pytest.raises(ValidationError):
        BaselineConfigurationRef(
            baseline_version_id="v",
            baseline_configuration_digest="A" * 64,
            artifact_type=ProfileArtifactType.ORCHESTRATION,
        )


def test_deep_immutability_and_input_alias_isolation() -> None:
    alias: dict[str, AdasJsonValue] = {"items": [{"k": "v"}], "mapping": {"x": 1}}
    payload = sample_record().model_dump(mode="python")
    payload["agent_contract_snapshot"] = alias
    record = AgentCandidateRecord.model_validate(payload)
    assert isinstance(record.agent_contract_snapshot, _FrozenAdasJsonObject)
    with pytest.raises(TypeError):
        record.agent_contract_snapshot["items"][0]["k"] = "mutated"  # type: ignore[index]
    alias_items = alias["items"]
    assert isinstance(alias_items, list)
    alias_mapping = alias_items[0]
    assert isinstance(alias_mapping, dict)
    alias_mapping["k"] = "external"
    assert canonical_record_json(record)["agent_contract_snapshot"]["items"][0]["k"] == "v"
    serialized = record.model_dump_json()
    import json as _json

    detached = _json.loads(serialized)
    detached["agent_contract_snapshot"]["mapping"]["x"] = 99
    assert canonical_record_json(record)["agent_contract_snapshot"]["mapping"]["x"] == 1


def test_evaluation_unknown_metrics_and_passed_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        AgentCandidateEvaluationResult(candidate_id="cand-1", passed=True)
    with pytest.raises(ValidationError):
        AgentCandidateEvaluationResult(
            candidate_id="cand-1",
            passed=True,
            quality_score=0.9,
            cost_normalized=0.1,
            latency_ms=10,
            total_tokens=100,
            utility_score=0.8,
            golden_pass_rate=1.5,
            negative_pass_rate=1.0,
            adversarial_pass_rate=1.0,
            execution_ref="exec-1",
            evaluation_ref="eval-1",
            evidence_complete=True,
        )
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
        baseline_execution_ref=AdasExecutionReference(execution_id="exec-base"),
        eval_run_ref=AdasEvalRunReference(eval_run_id="eval-run-1"),
        eval_comparison_ref=AdasEvalComparisonReference(comparison_id="cmp-1"),
        evidence_complete=True,
    )
    assert result.tool_calls is None


def test_evidence_digest_and_rejected_after_evaluation_preservation() -> None:
    record = sample_record(
        status=AgentCandidateStatus.REJECTED,
        rejection_reason="policy",
        static_gate_result={"passed": False, "reason": "rejected"},
    )
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
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        assembled_at=_NOW,
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    verify_evidence_bundle_digest(sealed)
    post_eval = AgentCandidateEvaluationResult(
        candidate_id=record.candidate_id,
        passed=False,
        failure_reasons=("cost",),
        execution_ref="exec-2",
        evaluation_ref="eval-2",
    )
    preserved = sealed.model_copy(update={"evaluation_result": post_eval})
    assert preserved.evaluation_result is not None
    with pytest.raises(ValueError):
        verify_evidence_bundle_digest(sealed.model_copy(update={"bundle_digest": "c" * 64}))


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
    with pytest.raises(ValidationError):
        AgentCandidateDraft(
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
            contract_draft={"tool": "deny"},
        )


def test_search_run_terminal_rules_and_unknown_costs() -> None:
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
    run = AgentDesignSearchRun(
        search_run_id="run-1",
        objective_id="obj-1",
        tenant_id="tenant-a",
        workspace_id="ws-a",
        task_class="t",
        envelope=sample_envelope(),
        strategy_name="rule",
        status="running",
        baseline=sample_baseline(),
        resource_accounting=ResourceAccountingEvidence(),
        started_at=_NOW,
    )
    assert run.promoted_candidate_id is None
    assert run.total_generation_cost is None


def test_q01_selection_outcome_separate_from_candidate_status() -> None:
    assert SearchSelectionOutcome.SELECT_CANDIDATE.value == "select_candidate"
    assert AgentCandidateStatus.PROMOTED.value == "promoted"


def test_q02_rejects_invalid_admitted_field() -> None:
    with pytest.raises(ValidationError):
        ConfigurationDelta(
            admitted_field="tools",
            prior_value=MergeStrategy.CONCAT,
            replacement_value=MergeStrategy.LAST_WINS,
        )


def test_q02_requires_human_approval() -> None:
    with pytest.raises(ValidationError):
        AgentDesignObjective(
            **{**sample_objective().model_dump(), "require_human_approval": False},
        )


def test_q02_min_utility_delta_default() -> None:
    assert sample_objective().min_utility_delta == 0.05


def test_q03_rejects_negative_generation_cost() -> None:
    with pytest.raises(ValidationError):
        AgentCandidateRecord(**{**sample_record().model_dump(), "generation_cost": -0.1})


def test_q03_rejects_whitespace_padded_digest() -> None:
    with pytest.raises(ValidationError):
        BaselineConfigurationRef(
            baseline_version_id="v",
            baseline_configuration_digest=f" {_DIGEST} ",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
        )


def test_q04_search_run_terminal_requires_structured_stop_reason() -> None:
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
            completed_at=_NOW + timedelta(seconds=1),
            stop_reason="done",
            selection_outcome=SearchSelectionOutcome.RETAIN_BASELINE,
        )


def test_q10_rejects_contradictory_bundle_configuration_digest() -> None:
    record = sample_record()
    with pytest.raises(ValidationError):
        AgentCandidateEvidenceBundle(
            bundle_id="bundle-1",
            candidate_id=record.candidate_id,
            objective_id=record.objective_id,
            tenant_id=record.tenant_id,
            workspace_id=record.workspace_id,
            created_at=_NOW,
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest="c" * 64,
            candidate_record=record,
            static_gate_result={"passed": True},
            assembled_at=_NOW,
        )


def test_q10_rejects_foreign_evaluation_candidate_id() -> None:
    record = sample_record()
    foreign_eval = AgentCandidateEvaluationResult(
        candidate_id="other-cand",
        passed=False,
        failure_reasons=("mismatch",),
    )
    with pytest.raises(ValidationError):
        AgentCandidateEvidenceBundle(
            bundle_id="bundle-1",
            candidate_id=record.candidate_id,
            objective_id=record.objective_id,
            tenant_id=record.tenant_id,
            workspace_id=record.workspace_id,
            created_at=_NOW,
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest=record.candidate_configuration_digest,
            candidate_record=record,
            evaluation_result=foreign_eval,
            static_gate_result={"passed": True},
            assembled_at=_NOW,
        )


def test_q12_evidence_digest_key_order_invariant() -> None:
    record = sample_record(
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        rejection_reason="x",
        static_gate_result={"passed": False, "reason": "x"},
    )
    gate_a: dict[str, AdasJsonValue] = {}
    gate_a["passed"] = False
    gate_a["reason"] = "x"
    gate_b: dict[str, AdasJsonValue] = {}
    gate_b["reason"] = "x"
    gate_b["passed"] = False
    common = {
        "bundle_id": "bundle-1",
        "candidate_id": record.candidate_id,
        "objective_id": record.objective_id,
        "tenant_id": record.tenant_id,
        "workspace_id": record.workspace_id,
        "created_at": _NOW,
        "created_by": "operator",
        "baseline_configuration_digest": _DIGEST,
        "candidate_configuration_digest": record.candidate_configuration_digest,
        "candidate_record": record,
        "terminal_evidence_outcome": W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        "assembled_at": _NOW,
        "sealed": True,
    }
    bundle_a = AgentCandidateEvidenceBundle(**common, static_gate_result=gate_a)
    bundle_b = AgentCandidateEvidenceBundle(**common, static_gate_result=gate_b)
    assert compute_evidence_bundle_digest(bundle_a) == compute_evidence_bundle_digest(bundle_b)
    instant = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    offset_equiv = instant.astimezone(UTC)
    bundle_t1 = AgentCandidateEvidenceBundle(
        **{**common, "static_gate_result": gate_a, "created_at": instant, "assembled_at": instant}
    )
    bundle_t2 = AgentCandidateEvidenceBundle(
        **{**common, "static_gate_result": gate_b, "created_at": offset_equiv, "assembled_at": offset_equiv}
    )
    assert compute_evidence_bundle_digest(bundle_t1) == compute_evidence_bundle_digest(bundle_t2)


def test_c02_duplicate_allowed_merge_alternatives_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate merge strategy"):
        AgentDesignObjective(
            **{
                **sample_objective().model_dump(),
                "allowed_merge_strategies": (
                    MergeStrategy.LAST_WINS,
                    MergeStrategy.CONCAT,
                    MergeStrategy.CONCAT,
                ),
            },
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("max_generation_cost", -1.0),
        ("max_generation_cost", float("inf")),
        ("max_eval_cost", -0.01),
        ("max_eval_cost", float("nan")),
    ],
)
def test_c02_invalid_optional_objective_budgets(field_name: str, bad_value: float) -> None:
    with pytest.raises(ValidationError):
        AgentDesignObjective(**{**sample_objective().model_dump(), field_name: bad_value})


def test_c02_mutable_json_wrapper_backing_blocked() -> None:
    payload = sample_record().model_dump(mode="python")
    payload["agent_contract_snapshot"] = {"marker": {"k": "v"}}
    record = AgentCandidateRecord.model_validate(payload)
    wrapper = record.agent_contract_snapshot
    assert isinstance(wrapper, _FrozenAdasJsonObject)
    with pytest.raises(TypeError):
        wrapper["marker"] = {"k": "z"}  # type: ignore[index]
    with pytest.raises(TypeError):
        setattr(wrapper, "marker", {})


def test_c02_invalid_bundle_digest_rejected_at_construction() -> None:
    record = sample_record(
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        rejection_reason="x",
        static_gate_result={"passed": False, "reason": "x"},
    )
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-x",
        candidate_id=record.candidate_id,
        objective_id=record.objective_id,
        tenant_id=record.tenant_id,
        workspace_id=record.workspace_id,
        created_at=_NOW,
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=record.candidate_configuration_digest,
        candidate_record=record,
        static_gate_result={"passed": False, "reason": "x"},
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        assembled_at=_NOW,
        sealed=True,
    )
    with pytest.raises(ValidationError, match="bundle_digest mismatch"):
        AgentCandidateEvidenceBundle(**{**bundle.model_dump(), "bundle_digest": "c" * 64})


def test_c02_search_run_mismatch_in_sealed_bundle() -> None:
    record = sample_record(
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        rejection_reason="x",
        static_gate_result={"passed": False, "reason": "x"},
    )
    with pytest.raises(ValidationError, match="search_run_id"):
        AgentCandidateEvidenceBundle(
            bundle_id="bundle-sr",
            candidate_id=record.candidate_id,
            objective_id=record.objective_id,
            tenant_id=record.tenant_id,
            workspace_id=record.workspace_id,
            search_run_id="other-run",
            created_at=_NOW,
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest=record.candidate_configuration_digest,
            candidate_record=record,
            static_gate_result={"passed": False, "reason": "x"},
            terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
            assembled_at=_NOW,
            sealed=True,
        )


def test_c02_incomplete_terminal_static_gate_shape() -> None:
    record = sample_record(status=AgentCandidateStatus.STATIC_GATE_FAILED, rejection_reason="x")
    with pytest.raises(ValidationError, match="passed"):
        AgentCandidateEvidenceBundle(
            bundle_id="bundle-shape",
            candidate_id=record.candidate_id,
            objective_id=record.objective_id,
            tenant_id=record.tenant_id,
            workspace_id=record.workspace_id,
            created_at=_NOW,
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest=record.candidate_configuration_digest,
            candidate_record=record,
            static_gate_result={"detail": "missing passed"},
            assembled_at=_NOW,
            sealed=True,
        )


def test_q03_rejects_negative_min_utility_delta() -> None:
    with pytest.raises(ValidationError):
        AgentDesignObjective(**{**sample_objective().model_dump(), "min_utility_delta": -0.01})


def test_q03_accepts_valid_signed_utility_delta_on_record() -> None:
    record = sample_record().model_copy(update={"utility_delta": -0.02})
    assert record.utility_delta == -0.02


@pytest.mark.parametrize(
    "status_value",
    [member.value for member in AgentCandidateStatus],
)
def test_q01_all_candidate_status_values(status_value: str) -> None:
    assert AgentCandidateStatus(status_value).value == status_value


@pytest.mark.parametrize(
    "outcome_value",
    [member.value for member in SearchSelectionOutcome],
)
def test_q01_selection_outcome_values(outcome_value: str) -> None:
    assert SearchSelectionOutcome(outcome_value).value == outcome_value
