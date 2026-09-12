# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence import (
    ADAPTIVE_INTELLIGENCE_TASK_ID,
    AdaptiveDecisionContextReference,
    AdaptiveDecisionIntelligenceEngine,
    AdaptiveDecisionIntelligenceInput,
    AdaptiveIntelligenceRunStatus,
    AdaptiveReasoningInsight,
    ConfidenceLevel,
    LifecycleHistoryContextProvider,
    RiskReasoningProvider,
    TechnicalRecommendationProvider,
    default_adaptive_decision_intelligence_engine,
    default_context_providers,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDataSourceKind,
    AdaptiveDataSourceRef,
    AdaptiveDecisionIntelligenceContext,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    OptimizationDataSourceKind,
    OptimizationDataSourceRef,
)
from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    OBSERVABILITY_TASK_ID,
    AnalysisPayloadKind,
    AnalyticsResultStatus,
    DecisionAnalyticsAuditMetadata,
    DecisionAnalyticsResult,
    GovernanceOutcomeCounts,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    ConfidenceLevel as OptimizationConfidenceLevel,
    DecisionOptimizationAuditMetadata,
    DecisionOptimizationResult,
    DecisionOptimizationSuggestion,
    OptimizationArea,
    OptimizationRunStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC)


def _decision_ref() -> AdaptiveDecisionContextReference:
    return AdaptiveDecisionContextReference(
        decision_id="new-decision-1",
        decision_subject="model_routing_choice",
    )


def _lifecycle_failed() -> DecisionLifecycleRecord:
    return DecisionLifecycleRecord(
        decision_id="prior-failed-1",
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=DecisionLifecycleState.FAILED,
        created_at=_stamp(),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.ORCHESTRATION,
                reference_id="orch-prior",
            ),
        ),
    )


def _governance_analytics() -> DecisionAnalyticsResult:
    return DecisionAnalyticsResult(
        observability_task_id=OBSERVABILITY_TASK_ID,
        status=AnalyticsResultStatus.COMPLETE,
        audit=DecisionAnalyticsAuditMetadata(
            analyzer_id="governance_outcome",
            analyzer_version="1",
            decision_ids=("prior-failed-1",),
            period_start=_stamp(),
            period_end=_stamp(),
        ),
        payload_kind=AnalysisPayloadKind.GOVERNANCE,
        governance_payload=GovernanceOutcomeCounts(
            allow=1,
            block=2,
            require_approval=0,
        ),
    )


def _optimization_result() -> DecisionOptimizationResult:
    return DecisionOptimizationResult(
        optimization_task_id="DS-E2E-15J-L9.DECISION-OPTIMIZATION-LEARNING-LOOP",
        status=OptimizationRunStatus.COMPLETE,
        audit=DecisionOptimizationAuditMetadata(
            engine_task_id="DS-E2E-15J-L9.DECISION-OPTIMIZATION-LEARNING-LOOP",
            engine_version="1",
            pattern_analyzer_ids=("test",),
            insight_generator_ids=("test",),
            recommendation_provider_ids=("test",),
            analyzed_at=_stamp(),
            analytics_result_refs=("governance_outcome:governance",),
            observation_decision_ids=("prior-failed-1",),
        ),
        patterns=(),
        insights=(),
        suggestions=(
            DecisionOptimizationSuggestion(
                suggestion_id="sug-1",
                source_decisions=("prior-failed-1",),
                optimization_area=OptimizationArea.GOVERNANCE_FRICTION,
                observation_references=(
                    OptimizationDataSourceRef(
                        source_kind=OptimizationDataSourceKind.ANALYTICS_RESULT,
                        reference_id="analytics:0",
                    ),
                ),
                rationale="Consider alternate model when blocks recur.",
                confidence=OptimizationConfidenceLevel.MEDIUM,
                generated_at=_stamp(),
                provider_id="human_review",
                provider_version="1",
                insight_ids=("ins-1",),
            ),
        ),
        aggregate_confidence=OptimizationConfidenceLevel.MEDIUM,
    )


def test_context_build_from_lifecycle_analytics_and_optimization() -> None:
    engine = default_adaptive_decision_intelligence_engine()
    input_data = AdaptiveDecisionIntelligenceInput(
        decision_context_reference=_decision_ref(),
        lifecycle_records=(_lifecycle_failed(),),
        analytics_results=(_governance_analytics(),),
        optimization_result=_optimization_result(),
    )

    context = engine.build_context(input_data)

    assert context.decision_context_reference == _decision_ref()
    provider_ids = {item.context_provider_id for item in context.historical_evidence}
    assert "lifecycle_history" in provider_ids
    assert "analytics_history" in provider_ids
    assert "optimization_context" in provider_ids
    assert any("failed" in item.summary.lower() for item in context.historical_evidence)
    assert context.data_source_refs


def test_reasoning_plugin_generates_insight() -> None:
    engine = AdaptiveDecisionIntelligenceEngine(
        context_providers=(LifecycleHistoryContextProvider(),),
        reasoning_providers=(RiskReasoningProvider(),),
        recommendation_providers=(),
    )
    input_data = AdaptiveDecisionIntelligenceInput(
        decision_context_reference=_decision_ref(),
        lifecycle_records=(_lifecycle_failed(),),
    )
    context = engine.build_context(input_data)

    insights = RiskReasoningProvider().reason(context)

    assert len(insights) == 1
    assert insights[0].reasoning_provider_id == "risk_reasoning"
    assert "risk" in insights[0].reasoning_summary.lower()


def test_recommendation_provider_from_reasoning_insight() -> None:
    reasoning = AdaptiveReasoningInsight(
        reasoning_insight_id="ri-1",
        reasoning_provider_id="risk_reasoning",
        reasoning_provider_version="1",
        reasoning_summary="Elevated risk from history.",
        source_evidence_ids=("lifecycle_history:prior-failed-1",),
        data_source_refs=(
            AdaptiveDataSourceRef(
                source_kind=AdaptiveDataSourceKind.LIFECYCLE_RECORD,
                reference_id="prior-failed-1",
            ),
        ),
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.8,
    )
    context = AdaptiveDecisionIntelligenceContext(
        decision_context_reference=_decision_ref(),
        historical_evidence=(),
        data_source_refs=(),
        context_provider_ids=("lifecycle_history",),
        context_provider_versions=("1",),
    )
    provider = TechnicalRecommendationProvider()
    recommendations = provider.recommend((reasoning,), context=context)

    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.recommendation_provider_id == "technical_recommendation"
    assert rec.recommendation_kind == "technical"
    assert rec.linked_reasoning_insight_ids == ("ri-1",)
    assert "automatic" not in rec.narrative.lower() or "human" in rec.narrative.lower()


class _StubReasoningProvider:
    provider_id = "stub_reasoning"
    provider_version = "stub-1"

    def reason(
        self,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveReasoningInsight, ...]:
        return (
            AdaptiveReasoningInsight(
                reasoning_insight_id="stub:ri",
                reasoning_provider_id=self.provider_id,
                reasoning_provider_version=self.provider_version,
                reasoning_summary="stub reasoning",
                source_evidence_ids=tuple(
                    item.evidence_id for item in context.historical_evidence
                ),
                data_source_refs=context.data_source_refs,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.2,
            ),
        )


def test_swappable_reasoning_plugin_without_engine_core_changes() -> None:
    input_data = AdaptiveDecisionIntelligenceInput(
        decision_context_reference=_decision_ref(),
        lifecycle_records=(_lifecycle_failed(),),
    )
    shared_context = default_context_providers()
    default_engine = AdaptiveDecisionIntelligenceEngine(
        context_providers=shared_context,
        reasoning_providers=(RiskReasoningProvider(),),
        recommendation_providers=(TechnicalRecommendationProvider(),),
    )
    stub_engine = AdaptiveDecisionIntelligenceEngine(
        context_providers=shared_context,
        reasoning_providers=(_StubReasoningProvider(),),
        recommendation_providers=(TechnicalRecommendationProvider(),),
    )

    default_result = default_engine.assist(input_data, run_at=_stamp())
    stub_result = stub_engine.assist(input_data, run_at=_stamp())

    assert (
        default_result.audit.context_provider_ids
        == stub_result.audit.context_provider_ids
    )
    assert (
        default_result.reasoning_insights[0].reasoning_provider_id == "risk_reasoning"
    )
    assert stub_result.reasoning_insights[0].reasoning_provider_id == "stub_reasoning"
    assert stub_result.reasoning_insights[0].reasoning_summary == "stub reasoning"


def test_missing_context_returns_controlled_result_without_error() -> None:
    engine = default_adaptive_decision_intelligence_engine()
    result = engine.assist(
        AdaptiveDecisionIntelligenceInput(
            decision_context_reference=_decision_ref(),
        ),
        run_at=_stamp(),
    )

    assert result.intelligence_task_id == ADAPTIVE_INTELLIGENCE_TASK_ID
    assert result.status is AdaptiveIntelligenceRunStatus.INSUFFICIENT_CONTEXT
    assert result.reasoning_insights == ()
    assert result.recommendations == ()
    assert result.insights == ()
