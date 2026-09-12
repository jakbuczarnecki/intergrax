# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R6.3 controlled execution integration guard."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.autonomy import (
    AutonomyAuditBundle,
    AutonomyControlDecision,
    AutonomyControlRequest,
    AutonomyDecisionContext,
    AutonomyEvaluationVerdict,
    AutonomyExecutionAdmissionContext,
    AutonomyExecutionAuthorizationStatus,
    AutonomyExecutionAuthorizingGuard,
    AutonomyExecutionGuard,
    AutonomyExecutionGuardRule,
    AutonomyExecutionGuardRuleVerdict,
    AutonomyGuardVerdict,
    AutonomyLevel,
    AutonomyPolicy,
    AutonomyPolicyOutcome,
    AutonomyRiskAssessment,
    AutonomyRiskBand,
    AutonomyRiskEvaluator,
    AutonomyRiskFactor,
    HumanApprovalRequirement,
    HumanApprovalRequirementResolver,
    mint_autonomy_control_decision_id,
    mint_autonomy_recommendation_correlation_id,
)
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.strategy_recommendation import (
    StrategyRecommendation,
    StrategyRecommendationBasisKind,
    StrategyRecommendationConfidenceLevel,
)
from intergrax.contracts.self_healing.strategy_recommendation.basis import StrategyRecommendationBasis
from intergrax.runtime.self_healing.autonomy import (
    AutonomyControlService,
    AutonomyDecisionEvaluationService,
    AutonomyPolicyPluginEvaluator,
    DefaultAutonomyExecutionGuard,
    DefaultHumanApprovalRequirementResolver,
    HumanApprovalPluginEvaluator,
    InMemoryAutonomyDecisionRepository,
    InMemoryAutonomyExecutionAuditRepository,
    PluginAutonomyControlEngine,
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


def _request() -> AutonomyControlRequest:
    correlation_id = mint_autonomy_recommendation_correlation_id()
    return AutonomyControlRequest(
        recommendation=_recommendation(),
        decision_context=AutonomyDecisionContext(
            recommendation_correlation_id=correlation_id,
            required_autonomy_level=AutonomyLevel.CONTROLLED_EXECUTION,
            audit=_audit(correlation_id),
        ),
    )


@dataclass(frozen=True, slots=True)
class ControlledExecutionPolicy:
    @property
    def policy_id(self) -> str:
        return "test.controlled_execution"

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyOutcome:
        _ = request
        return AutonomyPolicyOutcome(
            policy_id=self.policy_id,
            policy_version="1",
            suggested_level=AutonomyLevel.CONTROLLED_EXECUTION,
            constraint_descriptors=(),
            rationale="test controlled policy",
        )


@dataclass(frozen=True, slots=True)
class LowRiskEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "test.low_risk"

    def evaluate(self, request: AutonomyControlRequest, effective_level: AutonomyLevel) -> AutonomyRiskAssessment:
        _ = request
        _ = effective_level
        return AutonomyRiskAssessment(
            evaluator_id=self.evaluator_id,
            risk_band=AutonomyRiskBand.LOW,
            factors=(AutonomyRiskFactor(label="test", detail="low"),),
            rationale="low risk",
        )


def _controlled_stack() -> tuple[AutonomyControlService, AutonomyDecisionEvaluationService]:
    policy = ControlledExecutionPolicy()
    assert isinstance(policy, AutonomyPolicy)
    risk = LowRiskEvaluator()
    assert isinstance(risk, AutonomyRiskEvaluator)
    approval = DefaultHumanApprovalRequirementResolver()
    assert isinstance(approval, HumanApprovalRequirementResolver)
    control = AutonomyControlService(
        engine=PluginAutonomyControlEngine(
            policy=policy,
            risk_evaluator=risk,
            approval_resolver=approval,
        ),
    )
    evaluation = AutonomyDecisionEvaluationService(
        evaluator=PluginAutonomyDecisionEvaluator(
            policy_evaluator=AutonomyPolicyPluginEvaluator(policy),
            risk_evaluator=risk,
            approval_evaluator=HumanApprovalPluginEvaluator(approval),
        ),
    )
    return control, evaluation


def test_guard_authorization_success_when_evaluation_cleared() -> None:
    request = _request()
    control, evaluation_service = _controlled_stack()
    decision = control.evaluate(request)
    repo = InMemoryAutonomyDecisionRepository()
    evaluation = replace(evaluation_service, repository=repo).evaluate(request)
    assert evaluation.verdict is AutonomyEvaluationVerdict.CLEARED
    assert decision.auto_path_allowed is True

    admission = AutonomyExecutionAdmissionContext(
        tenant_id=_TENANT,
        recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        decision_id=decision.decision_id,
        prior_decision=decision,
    )
    guard = DefaultAutonomyExecutionGuard(evaluation_repository=repo)
    authorization = guard.authorize(admission)
    assert authorization.status is AutonomyExecutionAuthorizationStatus.AUTHORIZED
    assert authorization.permits_execution() is True
    check = guard.check(admission)
    assert check.verdict is AutonomyGuardVerdict.ALLOWED


def test_guard_denied_without_evaluation_fail_safe() -> None:
    request = _request()
    control, _ = _controlled_stack()
    decision = control.evaluate(request)
    admission = AutonomyExecutionAdmissionContext(
        tenant_id=_TENANT,
        recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        decision_id=decision.decision_id,
        prior_decision=decision,
    )
    guard = DefaultAutonomyExecutionGuard(evaluation_repository=InMemoryAutonomyDecisionRepository())
    authorization = guard.authorize(admission)
    assert authorization.status is AutonomyExecutionAuthorizationStatus.DENIED
    assert authorization.permits_execution() is False


def test_guard_denied_when_approval_required_without_token() -> None:
    request = _request()
    control, evaluation_service = _controlled_stack()
    baseline = control.evaluate(request)
    repo = InMemoryAutonomyDecisionRepository()
    evaluation = replace(evaluation_service, repository=repo).evaluate(request)

    @dataclass(frozen=True, slots=True)
    class RequireApprovalEvaluator:
        @property
        def evaluator_id(self) -> str:
            return "test.require_approval"

        def evaluate(
            self,
            request: AutonomyControlRequest,
            policy_result: object,
            risk_result: object,
            effective_level: AutonomyLevel,
        ) -> object:
            from intergrax.contracts.self_healing.autonomy.approval_evaluation import (
                HumanApprovalEvaluationResult,
            )

            _ = request
            _ = policy_result
            _ = risk_result
            _ = effective_level
            return HumanApprovalEvaluationResult(
                evaluator_id=self.evaluator_id,
                approval_required=True,
                reason_code="test.approval",
                rationale="approval required",
            )

    evaluation_service_override = AutonomyDecisionEvaluationService(
        evaluator=PluginAutonomyDecisionEvaluator(
            policy_evaluator=AutonomyPolicyPluginEvaluator(ControlledExecutionPolicy()),
            risk_evaluator=LowRiskEvaluator(),
            approval_evaluator=RequireApprovalEvaluator(),
        ),
        repository=repo,
    )
    evaluation_with_approval = evaluation_service_override.evaluate(request)
    assert evaluation_with_approval.approval_result.approval_required is True

    decision_id = mint_autonomy_control_decision_id()
    decision = AutonomyControlDecision(
        decision_id=decision_id,
        autonomy_level=AutonomyLevel.CONTROLLED_EXECUTION,
        auto_path_allowed=True,
        constraints=baseline.constraints,
        policy_outcome=baseline.policy_outcome,
        risk_outcome=baseline.risk_outcome,
        human_approval=HumanApprovalRequirement(
            required=False,
            reason_code="test.stale",
            rationale="stale decision snapshot for guard test",
        ),
        audit_bundle=baseline.audit_bundle,
        engine_id=baseline.engine_id,
        recommendation_correlation_id=baseline.recommendation_correlation_id,
    )

    admission = AutonomyExecutionAdmissionContext(
        tenant_id=_TENANT,
        recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        decision_id=decision_id,
        prior_decision=decision,
    )
    guard = DefaultAutonomyExecutionGuard(evaluation_repository=repo)
    authorization = guard.authorize(admission)
    assert authorization.status is AutonomyExecutionAuthorizationStatus.DENIED
    _ = evaluation


def test_guard_rule_plugin_can_deny() -> None:
    request = _request()
    control, evaluation_service = _controlled_stack()
    decision = control.evaluate(request)
    repo = InMemoryAutonomyDecisionRepository()
    replace(evaluation_service, repository=repo).evaluate(request)

    @dataclass(frozen=True, slots=True)
    class DenyRule:
        @property
        def rule_id(self) -> str:
            return "test.deny_rule"

        def assess(self, admission: object, evaluation: object) -> AutonomyExecutionGuardRuleVerdict:
            _ = admission
            _ = evaluation
            return AutonomyExecutionGuardRuleVerdict(
                permitted=False,
                rationale="plugin deny",
            )

    rule = DenyRule()
    assert isinstance(rule, AutonomyExecutionGuardRule)
    admission = AutonomyExecutionAdmissionContext(
        tenant_id=_TENANT,
        recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        decision_id=decision.decision_id,
        prior_decision=decision,
    )
    guard = DefaultAutonomyExecutionGuard(
        evaluation_repository=repo,
        guard_rules=(rule,),
    )
    assert guard.authorize(admission).status is AutonomyExecutionAuthorizationStatus.DENIED


def test_execution_audit_repository_records_guard_decision() -> None:
    request = _request()
    control, evaluation_service = _controlled_stack()
    decision = control.evaluate(request)
    repo = InMemoryAutonomyDecisionRepository()
    audit_repo = InMemoryAutonomyExecutionAuditRepository()
    replace(evaluation_service, repository=repo).evaluate(request)
    admission = AutonomyExecutionAdmissionContext(
        tenant_id=_TENANT,
        recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        decision_id=decision.decision_id,
        prior_decision=decision,
    )
    guard = DefaultAutonomyExecutionGuard(
        evaluation_repository=repo,
        audit_repository=audit_repo,
    )
    guard.authorize(admission)
    assert len(audit_repo.records) == 1
    record = audit_repo.records[0]
    assert record.decision_id == decision.decision_id
    assert record.evaluation_id is not None


def test_default_guard_satisfies_guard_protocols() -> None:
    guard = DefaultAutonomyExecutionGuard(
        evaluation_repository=InMemoryAutonomyDecisionRepository(),
    )
    assert isinstance(guard, AutonomyExecutionGuard)
    assert isinstance(guard, AutonomyExecutionAuthorizingGuard)


def test_guard_runtime_module_has_no_executor_coupling() -> None:
    module_names = (
        "intergrax.runtime.self_healing.autonomy.default_execution_guard",
        "intergrax.runtime.self_healing.autonomy.guard_support",
    )
    forbidden = ("execution_engine", "SelfHealingActionProvider", "UAEPExecutor")
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden:
            assert token.lower() not in lowered
