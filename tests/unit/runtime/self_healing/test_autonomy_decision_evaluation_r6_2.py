# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R6.2 autonomy decision evaluation."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.autonomy import (
    AUTONOMY_EVALUATION_CONTRACT_VERSION,
    AutonomyAuditBundle,
    AutonomyDecisionContext,
    AutonomyDecisionCorrelationQuery,
    AutonomyDecisionEvaluator,
    AutonomyDecisionRepository,
    AutonomyEvaluationAuditRecorder,
    AutonomyEvaluationAuditTrailEntry,
    AutonomyEvaluationVerdict,
    AutonomyLevel,
    AutonomyPolicyEvaluationVerdict,
    AutonomyPolicyEvaluator,
    AutonomyRiskAssessment,
    AutonomyRiskBand,
    AutonomyRiskFactor,
    HumanApprovalEvaluator,
    mint_autonomy_recommendation_correlation_id,
)
from intergrax.contracts.self_healing.autonomy.evaluation_confidence import AutonomyEvaluationConfidence
from intergrax.contracts.self_healing.autonomy.evaluation_confidence import (
    AutonomyEvaluationConfidenceLevel,
)
from intergrax.contracts.self_healing.autonomy.policy_evaluation import AutonomyPolicyEvaluationResult
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import autonomy_risk_evaluation_result_from_assessment
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.strategy_recommendation import (
    StrategyRecommendation,
    StrategyRecommendationBasisKind,
    StrategyRecommendationConfidenceLevel,
)
from intergrax.contracts.self_healing.strategy_recommendation.basis import StrategyRecommendationBasis
from intergrax.runtime.self_healing.autonomy import (
    AutonomyDecisionEvaluationService,
    AutonomyPolicyPluginEvaluator,
    DefaultAutonomyPolicy,
    DefaultAutonomyRiskEvaluator,
    DefaultHumanApprovalRequirementResolver,
    HumanApprovalPluginEvaluator,
    InMemoryAutonomyDecisionRepository,
    PluginAutonomyDecisionEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_INVESTIGATION = "inv-db"
_PROBLEM = "problem.db"
_STRATEGY = "platform.database.retry"


def _assessment() -> StrategyQualityAssessment:
    return StrategyQualityAssessment(
        tenant_id=_TENANT,
        strategy_id=_STRATEGY,
        execution_count=2,
        successful_executions=2,
        failed_executions=0,
        rolled_back_executions=0,
        inconclusive_executions=0,
        success_ratio=1.0,
        average_recovery_time_seconds=3.0,
        min_recovery_time_seconds=2.0,
        max_recovery_time_seconds=4.0,
        quality_score=0.9,
        evaluator_id="test.evaluator",
        evidence_refs=("evidence://1",),
    )


def _recommendation() -> StrategyRecommendation:
    basis = StrategyRecommendationBasis(
        kind=StrategyRecommendationBasisKind.HIGHEST_HISTORICAL_QUALITY_SCORE,
        summary="test basis",
    )
    return StrategyRecommendation(
        tenant_id=_TENANT,
        diagnostic_investigation_id=_INVESTIGATION,
        problem_id=_PROBLEM,
        recommended_strategy_id=_STRATEGY,
        ranked_strategy_ids=(_STRATEGY,),
        basis=basis,
        confidence=StrategyRecommendationConfidenceLevel.HIGH,
        supporting_assessment=_assessment(),
        engine_id="test.recommendation_engine",
    )


def _audit(correlation_id: str) -> AutonomyAuditBundle:
    return AutonomyAuditBundle(
        tenant_id=_TENANT,
        diagnostic_investigation_id=_INVESTIGATION,
        problem_id=_PROBLEM,
        recommendation_correlation_id=correlation_id,
        recorded_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
    )


def _request(
    *,
    required_level: AutonomyLevel = AutonomyLevel.RECOMMEND_ONLY,
) -> AutonomyControlRequest:
    correlation_id = mint_autonomy_recommendation_correlation_id()
    return AutonomyControlRequest(
        recommendation=_recommendation(),
        decision_context=AutonomyDecisionContext(
            recommendation_correlation_id=correlation_id,
            required_autonomy_level=required_level,
            audit=_audit(correlation_id),
        ),
    )


def _evaluation_service() -> AutonomyDecisionEvaluationService:
    evaluator = PluginAutonomyDecisionEvaluator(
        policy_evaluator=AutonomyPolicyPluginEvaluator(DefaultAutonomyPolicy()),
        risk_evaluator=DefaultAutonomyRiskEvaluator(),
        approval_evaluator=HumanApprovalPluginEvaluator(
            DefaultHumanApprovalRequirementResolver(),
        ),
    )
    return AutonomyDecisionEvaluationService(evaluator=evaluator)


def test_evaluation_result_contains_explainable_fields() -> None:
    result = _evaluation_service().evaluate(_request())
    assert result.evaluation_id.startswith("sh_aut_eval_")
    assert result.contract_version == AUTONOMY_EVALUATION_CONTRACT_VERSION
    assert result.verdict is AutonomyEvaluationVerdict.CLEARED
    assert result.autonomy_level is AutonomyLevel.RECOMMEND_ONLY
    assert result.confidence.score == pytest.approx(0.75)
    assert result.confidence.level is AutonomyEvaluationConfidenceLevel.MEDIUM
    assert result.explanation.because
    assert any(bullet.code == "policy.permits_posture" for bullet in result.explanation.because)


def test_policy_evaluation_result_separate_from_final_verdict() -> None:
    result = _evaluation_service().evaluate(
        _request(required_level=AutonomyLevel.CONTROLLED_EXECUTION),
    )
    assert result.policy_result.verdict is AutonomyPolicyEvaluationVerdict.FAIL
    assert result.verdict is AutonomyEvaluationVerdict.DENIED


def test_risk_evaluation_result_wraps_assessment() -> None:
    assessment = DefaultAutonomyRiskEvaluator().evaluate(
        _request(),
        AutonomyLevel.RECOMMEND_ONLY,
    )
    confidence = AutonomyEvaluationConfidence(
        level=AutonomyEvaluationConfidenceLevel.MEDIUM,
        score=0.6,
    )
    risk_result = autonomy_risk_evaluation_result_from_assessment(assessment, confidence)
    assert risk_result.risk_band is AutonomyRiskBand.UNKNOWN
    assert risk_result.assessment.evaluator_id == assessment.evaluator_id


def test_approval_evaluation_reflects_human_gate() -> None:
    @dataclass(frozen=True, slots=True)
    class HighRiskEvaluator:
        @property
        def evaluator_id(self) -> str:
            return "test.high_risk"

        def evaluate(self, request: object, effective_level: AutonomyLevel) -> AutonomyRiskAssessment:
            _ = request
            _ = effective_level
            return AutonomyRiskAssessment(
                evaluator_id=self.evaluator_id,
                risk_band=AutonomyRiskBand.HIGH,
                factors=(AutonomyRiskFactor(label="test", detail="elevated"),),
                rationale="test elevated risk",
            )

    evaluator = PluginAutonomyDecisionEvaluator(
        policy_evaluator=AutonomyPolicyPluginEvaluator(DefaultAutonomyPolicy()),
        risk_evaluator=HighRiskEvaluator(),
        approval_evaluator=HumanApprovalPluginEvaluator(
            DefaultHumanApprovalRequirementResolver(),
        ),
    )
    result = AutonomyDecisionEvaluationService(evaluator=evaluator).evaluate(_request())
    assert result.approval_result.approval_required is True
    assert result.verdict is AutonomyEvaluationVerdict.DENIED


def test_decision_evaluator_plugin_is_swappable() -> None:
    @dataclass(frozen=True, slots=True)
    class AlwaysDenyPolicyEvaluator:
        @property
        def evaluator_id(self) -> str:
            return "test.always_deny_policy"

        def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyEvaluationResult:
            _ = request
            return AutonomyPolicyEvaluationResult(
                evaluator_id=self.evaluator_id,
                policy_id="test.policy",
                policy_version="1",
                verdict=AutonomyPolicyEvaluationVerdict.FAIL,
                suggested_level=AutonomyLevel.OBSERVE_ONLY,
                constraint_descriptors=(),
                rationale="deny for test",
            )

    evaluator = PluginAutonomyDecisionEvaluator(
        policy_evaluator=AlwaysDenyPolicyEvaluator(),
        risk_evaluator=DefaultAutonomyRiskEvaluator(),
        approval_evaluator=HumanApprovalPluginEvaluator(
            DefaultHumanApprovalRequirementResolver(),
        ),
    )
    assert isinstance(evaluator, AutonomyDecisionEvaluator)
    result = AutonomyDecisionEvaluationService(evaluator=evaluator).evaluate(_request())
    assert result.verdict is AutonomyEvaluationVerdict.DENIED


def test_repository_port_persists_evaluation_results() -> None:
    repo = InMemoryAutonomyDecisionRepository()
    service = AutonomyDecisionEvaluationService(evaluator=_evaluation_service().evaluator, repository=repo)
    request = _request()
    result = service.evaluate(request)
    stored = repo.get_latest_evaluation(
        AutonomyDecisionCorrelationQuery(
            tenant_id=_TENANT,
            recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        ),
    )
    assert stored is not None
    assert stored.evaluation_id == result.evaluation_id
    assert isinstance(repo, AutonomyDecisionRepository)


def test_audit_trail_entry_from_evaluation_result() -> None:
    @dataclass
    class RecordingAudit:
        entries: list[AutonomyEvaluationAuditTrailEntry] = field(default_factory=list)

        def record(self, entry: AutonomyEvaluationAuditTrailEntry) -> None:
            self.entries.append(entry)

    audit = RecordingAudit()
    service = AutonomyDecisionEvaluationService(
        evaluator=_evaluation_service().evaluator,
        audit_recorder=audit,
    )
    result = service.evaluate(_request())
    assert len(audit.entries) == 1
    entry = audit.entries[0]
    assert entry.evaluation_id == result.evaluation_id
    assert entry.policy_id == result.policy_result.policy_id
    assert isinstance(audit, AutonomyEvaluationAuditRecorder)


def test_r6_2_runtime_has_no_execution_coupling() -> None:
    forbidden_tokens = (
        "lifecycle",
        "orchestrator",
        "SelfHealingActionProvider",
        "execution_engine",
        "AutonomyExecutionGuard",
    )
    module_names = (
        "intergrax.runtime.self_healing.autonomy.decision_evaluation_service",
        "intergrax.runtime.self_healing.autonomy.plugin_decision_evaluator",
        "intergrax.runtime.self_healing.autonomy.evaluation_support",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden_tokens:
            assert token.lower() not in lowered


def test_policy_and_approval_evaluators_are_runtime_checkable() -> None:
    policy_evaluator = AutonomyPolicyPluginEvaluator(DefaultAutonomyPolicy())
    approval_evaluator = HumanApprovalPluginEvaluator(DefaultHumanApprovalRequirementResolver())
    assert isinstance(policy_evaluator, AutonomyPolicyEvaluator)
    assert isinstance(approval_evaluator, HumanApprovalEvaluator)
