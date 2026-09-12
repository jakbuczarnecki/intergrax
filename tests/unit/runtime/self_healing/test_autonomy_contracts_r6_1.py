# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R6.1 autonomy contracts foundation."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.autonomy import (
    AutonomyAuditBundle,
    AutonomyControlDecision,
    AutonomyControlEngine,
    AutonomyControlRequest,
    AutonomyDecisionContext,
    AutonomyDecisionCorrelationQuery,
    AutonomyExecutionAdmissionContext,
    AutonomyExecutionGuard,
    AutonomyGuardVerdict,
    AutonomyLevel,
    AutonomyPolicy,
    AutonomyPolicyOutcome,
    AutonomyRepository,
    AutonomyRiskAssessment,
    AutonomyRiskBand,
    AutonomyRiskEvaluator,
    AutonomyRiskFactor,
    HumanApprovalRequirement,
    HumanApprovalRequirementResolver,
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
    DefaultAutonomyPolicy,
    DefaultAutonomyRiskEvaluator,
    DefaultHumanApprovalRequirementResolver,
    InMemoryAutonomyRepository,
    PluginAutonomyControlEngine,
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
    required_level: AutonomyLevel = AutonomyLevel.CONTROLLED_EXECUTION,
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


def _service() -> AutonomyControlService:
    engine = PluginAutonomyControlEngine(
        policy=DefaultAutonomyPolicy(),
        risk_evaluator=DefaultAutonomyRiskEvaluator(),
        approval_resolver=DefaultHumanApprovalRequirementResolver(),
    )
    return AutonomyControlService(engine=engine)


def test_autonomy_levels_include_inactive_full_autonomy_contract() -> None:
    assert AutonomyLevel.FULL_AUTONOMY.value == "FULL_AUTONOMY"
    with pytest.raises(ValueError, match="FULL_AUTONOMY"):
        AutonomyLevel.FULL_AUTONOMY.ensure_runtime_activatable()


def test_decision_context_rejects_full_autonomy_required_level() -> None:
    correlation_id = mint_autonomy_recommendation_correlation_id()
    with pytest.raises(ValueError, match="FULL_AUTONOMY"):
        AutonomyDecisionContext(
            recommendation_correlation_id=correlation_id,
            required_autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            audit=_audit(correlation_id),
        )


def test_default_policy_is_recommend_only() -> None:
    outcome = DefaultAutonomyPolicy().evaluate(_request())
    assert outcome.suggested_level is AutonomyLevel.RECOMMEND_ONLY
    assert outcome.suggested_level is not AutonomyLevel.CONTROLLED_EXECUTION


def test_plugin_engine_merges_policy_conservatively() -> None:
    decision = _service().evaluate(_request(required_level=AutonomyLevel.CONTROLLED_EXECUTION))
    assert decision.autonomy_level is AutonomyLevel.RECOMMEND_ONLY
    assert decision.auto_path_allowed is False
    assert decision.policy_outcome.policy_id == "platform.default_autonomy"


def test_default_risk_evaluator_returns_neutral_band() -> None:
    assessment = DefaultAutonomyRiskEvaluator().evaluate(
        _request(),
        AutonomyLevel.RECOMMEND_ONLY,
    )
    assert assessment.risk_band is AutonomyRiskBand.UNKNOWN


def test_approval_resolver_requires_human_for_elevated_risk() -> None:
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

    engine = PluginAutonomyControlEngine(
        policy=DefaultAutonomyPolicy(),
        risk_evaluator=HighRiskEvaluator(),
        approval_resolver=DefaultHumanApprovalRequirementResolver(),
    )
    decision = AutonomyControlService(engine=engine).evaluate(_request())
    assert decision.human_approval.required is True


def test_policy_plugin_is_swappable() -> None:
    @dataclass(frozen=True, slots=True)
    class ApprovalOnlyPolicy:
        @property
        def policy_id(self) -> str:
            return "test.approval_only"

        def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyOutcome:
            _ = request
            return AutonomyPolicyOutcome(
                policy_id=self.policy_id,
                policy_version="1",
                suggested_level=AutonomyLevel.APPROVAL_REQUIRED,
                constraint_descriptors=(),
                rationale="test policy",
            )

    policy = ApprovalOnlyPolicy()
    assert isinstance(policy, AutonomyPolicy)
    engine = PluginAutonomyControlEngine(
        policy=policy,
        risk_evaluator=DefaultAutonomyRiskEvaluator(),
        approval_resolver=DefaultHumanApprovalRequirementResolver(),
    )
    decision = AutonomyControlService(engine=engine).evaluate(_request())
    assert decision.autonomy_level is AutonomyLevel.APPROVAL_REQUIRED
    assert decision.human_approval.required is True


def test_repository_port_persists_decisions() -> None:
    repo = InMemoryAutonomyRepository()
    service = AutonomyControlService(engine=_service().engine, repository=repo)
    request = _request()
    decision = service.evaluate(request)
    stored = repo.get_latest_by_correlation(
        AutonomyDecisionCorrelationQuery(
            tenant_id=_TENANT,
            recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        ),
    )
    assert stored is not None
    assert stored.decision_id == decision.decision_id
    assert isinstance(repo, AutonomyRepository)


def test_execution_guard_contract_without_runtime_wiring() -> None:
    @dataclass(frozen=True, slots=True)
    class DenyGuard:
        @property
        def guard_id(self) -> str:
            return "test.deny_guard"

        def check(self, admission: AutonomyExecutionAdmissionContext) -> object:
            from intergrax.contracts.self_healing.autonomy.guard import AutonomyGuardCheckResult

            _ = admission
            return AutonomyGuardCheckResult(
                verdict=AutonomyGuardVerdict.DENIED,
                rationale="guard contract only",
            )

    guard = DenyGuard()
    assert isinstance(guard, AutonomyExecutionGuard)


def test_policy_outcome_rejects_full_autonomy_suggestion() -> None:
    with pytest.raises(ValueError, match="FULL_AUTONOMY"):
        AutonomyPolicyOutcome(
            policy_id="bad",
            policy_version="1",
            suggested_level=AutonomyLevel.FULL_AUTONOMY,
            constraint_descriptors=(),
            rationale="invalid",
        )


def test_autonomy_domain_contracts_have_no_runtime_imports() -> None:
    modules = (
        "intergrax.contracts.self_healing.autonomy.level",
        "intergrax.contracts.self_healing.autonomy.engine",
        "intergrax.contracts.self_healing.autonomy.policy",
        "intergrax.contracts.self_healing.autonomy.guard",
        "intergrax.contracts.self_healing.autonomy.repository",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = inspect.getfile(module)
        assert "intergrax\\runtime" not in source_path
        assert "intergrax/runtime" not in source_path


def test_autonomy_runtime_has_no_execution_coupling() -> None:
    forbidden_tokens = (
        "lifecycle",
        "orchestrator",
        "SelfHealingActionProvider",
        "execution_engine",
        "HighestConfidenceStrategySelector",
    )
    module_names = (
        "intergrax.runtime.self_healing.autonomy.service",
        "intergrax.runtime.self_healing.autonomy.plugin_control_engine",
        "intergrax.runtime.self_healing.autonomy.default_policy",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden_tokens:
            assert token.lower() not in lowered


def test_risk_and_approval_plugins_are_runtime_checkable() -> None:
    assert isinstance(DefaultAutonomyRiskEvaluator(), AutonomyRiskEvaluator)
    assert isinstance(DefaultHumanApprovalRequirementResolver(), HumanApprovalRequirementResolver)
    engine = PluginAutonomyControlEngine(
        policy=DefaultAutonomyPolicy(),
        risk_evaluator=DefaultAutonomyRiskEvaluator(),
        approval_resolver=DefaultHumanApprovalRequirementResolver(),
    )
    assert isinstance(engine, AutonomyControlEngine)


def test_control_decision_rejects_auto_path_without_controlled_level() -> None:
    policy_outcome = DefaultAutonomyPolicy().evaluate(_request())
    risk = DefaultAutonomyRiskEvaluator().evaluate(_request(), AutonomyLevel.RECOMMEND_ONLY)
    approval = HumanApprovalRequirement(
        required=False,
        reason_code="test",
        rationale="test",
    )
    request = _request()
    with pytest.raises(ValueError, match="auto_path_allowed requires"):
        AutonomyControlDecision(
            decision_id="sh_aut_dec_test",
            autonomy_level=AutonomyLevel.RECOMMEND_ONLY,
            auto_path_allowed=True,
            constraints=(),
            policy_outcome=policy_outcome,
            risk_outcome=risk,
            human_approval=approval,
            audit_bundle=request.decision_context.audit,
            engine_id="test",
            recommendation_correlation_id=request.decision_context.recommendation_correlation_id,
        )
