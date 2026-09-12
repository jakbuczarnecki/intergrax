# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R6.3 execution spine integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.autonomy import (
    AutonomyAuditBundle,
    AutonomyControlRequest,
    AutonomyDecisionContext,
    AutonomyExecutionAdmissionContext,
    AutonomyExecutionDeniedError,
    AutonomyLevel,
    AutonomyPolicy,
    AutonomyPolicyOutcome,
    AutonomyRiskAssessment,
    AutonomyRiskBand,
    AutonomyRiskEvaluator,
    AutonomyRiskFactor,
    mint_autonomy_recommendation_correlation_id,
)
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.strategy_recommendation import (
    StrategyRecommendation,
    StrategyRecommendationBasisKind,
    StrategyRecommendationConfidenceLevel,
)
from intergrax.contracts.self_healing.strategy_recommendation.basis import StrategyRecommendationBasis
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.self_healing.autonomy import (
    AutonomyControlService,
    AutonomyDecisionEvaluationService,
    AutonomyPolicyPluginEvaluator,
    DefaultAutonomyExecutionBoundary,
    DefaultAutonomyExecutionGuard,
    DefaultHumanApprovalRequirementResolver,
    HumanApprovalPluginEvaluator,
    InMemoryAutonomyDecisionRepository,
    PluginAutonomyControlEngine,
    PluginAutonomyDecisionEvaluator,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT = "tenant-a"
_INVESTIGATION = "inv-db"
_PROBLEM = "problem.db"
_STRATEGY = "platform.database.retry"


@dataclass(frozen=True, slots=True)
class WorkRequest:
    payload: str
    admission: AutonomyExecutionAdmissionContext | None = None


@dataclass(frozen=True, slots=True)
class WorkResult:
    payload: str


@dataclass
class CountingDelegate:
    call_count: int = 0
    last_request: WorkRequest | None = None

    async def execute(self, request: WorkRequest) -> WorkResult:
        self.call_count += 1
        self.last_request = request
        return WorkResult(payload=request.payload)


@dataclass(frozen=True, slots=True)
class RequestAdmissionSource:
    def admission_for(self, request: WorkRequest) -> AutonomyExecutionAdmissionContext | None:
        return request.admission


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


def _prepare_admission() -> AutonomyExecutionAdmissionContext:
    correlation_id = mint_autonomy_recommendation_correlation_id()
    request = AutonomyControlRequest(
        recommendation=_recommendation(),
        decision_context=AutonomyDecisionContext(
            recommendation_correlation_id=correlation_id,
            required_autonomy_level=AutonomyLevel.CONTROLLED_EXECUTION,
            audit=_audit(correlation_id),
        ),
    )
    policy = ControlledExecutionPolicy()
    assert isinstance(policy, AutonomyPolicy)
    risk = LowRiskEvaluator()
    assert isinstance(risk, AutonomyRiskEvaluator)
    approval = DefaultHumanApprovalRequirementResolver()
    control = AutonomyControlService(
        engine=PluginAutonomyControlEngine(
            policy=policy,
            risk_evaluator=risk,
            approval_resolver=approval,
        ),
    )
    decision = control.evaluate(request)
    repo = InMemoryAutonomyDecisionRepository()
    evaluation_service = AutonomyDecisionEvaluationService(
        evaluator=PluginAutonomyDecisionEvaluator(
            policy_evaluator=AutonomyPolicyPluginEvaluator(policy),
            risk_evaluator=risk,
            approval_evaluator=HumanApprovalPluginEvaluator(approval),
        ),
        repository=repo,
    )
    evaluation_service.evaluate(request)
    return AutonomyExecutionAdmissionContext(
        tenant_id=_TENANT,
        recommendation_correlation_id=correlation_id,
        decision_id=decision.decision_id,
        prior_decision=decision,
    )


@pytest.mark.asyncio
async def test_legacy_request_without_admission_skips_guard() -> None:
    delegate = CountingDelegate()
    boundary = ExecutionBoundary[WorkRequest, WorkResult](delegate)
    result = await boundary.execute(WorkRequest(payload="legacy"))
    assert result.payload == "legacy"
    assert delegate.call_count == 1


@pytest.mark.asyncio
async def test_denied_autonomy_blocks_delegate_before_execution() -> None:
    delegate = CountingDelegate()
    repo = InMemoryAutonomyDecisionRepository()
    guard = DefaultAutonomyExecutionGuard(evaluation_repository=repo)
    spine_boundary = DefaultAutonomyExecutionBoundary(guard=guard)
    hook = spine_boundary.spine_admission_hook(RequestAdmissionSource())
    execution = ExecutionBoundary[WorkRequest, WorkResult](
        delegate,
        admission_hooks=(hook,),
    )
    admission = _prepare_admission()
    with pytest.raises(AutonomyExecutionDeniedError):
        await execution.execute(WorkRequest(payload="blocked", admission=admission))
    assert delegate.call_count == 0


@pytest.mark.asyncio
async def test_authorized_autonomy_uses_existing_execution_flow() -> None:
    delegate = CountingDelegate()
    admission = _prepare_admission()
    repo = InMemoryAutonomyDecisionRepository()
    policy = ControlledExecutionPolicy()
    risk = LowRiskEvaluator()
    correlation_id = admission.recommendation_correlation_id
    control_request = AutonomyControlRequest(
        recommendation=_recommendation(),
        decision_context=AutonomyDecisionContext(
            recommendation_correlation_id=correlation_id,
            required_autonomy_level=AutonomyLevel.CONTROLLED_EXECUTION,
            audit=_audit(correlation_id),
        ),
    )
    evaluation_service = AutonomyDecisionEvaluationService(
        evaluator=PluginAutonomyDecisionEvaluator(
            policy_evaluator=AutonomyPolicyPluginEvaluator(policy),
            risk_evaluator=risk,
            approval_evaluator=HumanApprovalPluginEvaluator(DefaultHumanApprovalRequirementResolver()),
        ),
        repository=repo,
    )
    evaluation_service.evaluate(control_request)

    guard = DefaultAutonomyExecutionGuard(evaluation_repository=repo)
    spine_boundary = DefaultAutonomyExecutionBoundary(guard=guard)
    hook = spine_boundary.spine_admission_hook(RequestAdmissionSource())
    execution = ExecutionBoundary[WorkRequest, WorkResult](
        delegate,
        admission_hooks=(hook,),
    )
    result = await execution.execute(WorkRequest(payload="ok", admission=admission))
    assert result.payload == "ok"
    assert delegate.call_count == 1
