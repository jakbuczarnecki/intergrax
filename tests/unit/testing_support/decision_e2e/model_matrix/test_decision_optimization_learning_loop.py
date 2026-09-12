# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    OBSERVABILITY_TASK_ID,
    AnalysisPayloadKind,
    AnalyticsResultStatus,
    DecisionAnalyticsAuditMetadata,
    DecisionAnalyticsResult,
    GovernanceOutcomeCounts,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop import (
    OPTIMIZATION_TASK_ID,
    ConfidenceLevel,
    DecisionOptimizationContext,
    DecisionOptimizationEngine,
    DecisionOptimizationSuggestion,
    GovernanceFrictionPatternAnalyzer,
    HumanReviewRecommendationProvider,
    OptimizationArea,
    OptimizationInsight,
    OptimizationRunStatus,
    PatternLinkageInsightGenerator,
    default_decision_optimization_engine,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    DetectedOptimizationPattern,
    OptimizationDataSourceKind,
    OptimizationDataSourceRef,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)


def _governance_analytics(
    *,
    allow: int,
    block: int,
    require_approval: int,
    decision_ids: tuple[str, ...] = ("d-gov-1", "d-gov-2"),
) -> DecisionAnalyticsResult:
    return DecisionAnalyticsResult(
        observability_task_id=OBSERVABILITY_TASK_ID,
        status=AnalyticsResultStatus.COMPLETE,
        audit=DecisionAnalyticsAuditMetadata(
            analyzer_id="governance_outcome",
            analyzer_version="1",
            decision_ids=decision_ids,
            period_start=_stamp(),
            period_end=_stamp(),
        ),
        payload_kind=AnalysisPayloadKind.GOVERNANCE,
        governance_payload=GovernanceOutcomeCounts(
            allow=allow,
            block=block,
            require_approval=require_approval,
        ),
    )


def test_pattern_detection_from_analytics_produces_insight() -> None:
    context = DecisionOptimizationContext(
        analytics_results=(
            _governance_analytics(allow=1, block=3, require_approval=1),
        ),
    )
    engine = default_decision_optimization_engine(
        insight_generators=(PatternLinkageInsightGenerator(),),
        recommendation_providers=(),
    )

    result = engine.run(context, run_at=_stamp())

    assert result.optimization_task_id == OPTIMIZATION_TASK_ID
    assert result.patterns
    assert any(
        item.optimization_area is OptimizationArea.GOVERNANCE_FRICTION
        for item in result.patterns
    )
    assert result.insights
    assert result.insights[0].generator_id == "pattern_linkage"
    assert result.insights[0].pattern_ids == (result.patterns[0].pattern_id,)


def test_insight_yields_human_review_suggestion() -> None:
    pattern = DetectedOptimizationPattern(
        pattern_id="test-pattern-1",
        analyzer_id="test_analyzer",
        analyzer_version="1",
        optimization_area=OptimizationArea.GOVERNANCE_FRICTION,
        summary="test friction",
        source_decision_ids=("dec-1",),
        data_source_refs=(
            OptimizationDataSourceRef(
                source_kind=OptimizationDataSourceKind.ANALYTICS_RESULT,
                reference_id="analytics:0",
            ),
        ),
        confidence=ConfidenceLevel.HIGH,
    )
    insight = OptimizationInsight(
        insight_id="insight-1",
        generator_id="test_gen",
        generator_version="1",
        optimization_area=OptimizationArea.GOVERNANCE_FRICTION,
        narrative="Friction observed.",
        pattern_ids=(pattern.pattern_id,),
        source_decision_ids=pattern.source_decision_ids,
        data_source_refs=pattern.data_source_refs,
        confidence=ConfidenceLevel.HIGH,
    )
    provider = HumanReviewRecommendationProvider()
    suggestions = provider.recommend(
        (insight,),
        context=DecisionOptimizationContext(analytics_results=()),
        generated_at=_stamp(),
    )

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.source_decisions == ("dec-1",)
    assert suggestion.optimization_area is OptimizationArea.GOVERNANCE_FRICTION
    assert suggestion.confidence is ConfidenceLevel.HIGH
    assert "human" in suggestion.rationale.lower() or "Consider" in suggestion.rationale
    assert suggestion.insight_ids == ("insight-1",)


class _TestFrictionAnalyzer:
    analyzer_id = "test_friction_plugin"
    analyzer_version = "test-1"

    def detect_patterns(
        self,
        context: DecisionOptimizationContext,
    ) -> tuple[DetectedOptimizationPattern, ...]:
        if not context.analytics_results:
            return ()
        return (
            DetectedOptimizationPattern(
                pattern_id="plugin-pattern",
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                optimization_area=OptimizationArea.CUSTOM,
                summary="plugin_ok",
                source_decision_ids=("plugin-dec",),
                data_source_refs=(
                    OptimizationDataSourceRef(
                        source_kind=OptimizationDataSourceKind.ANALYTICS_RESULT,
                        reference_id="analytics:plugin",
                    ),
                ),
                confidence=ConfidenceLevel.LOW,
            ),
        )


def test_plugin_pattern_analyzer_without_engine_core_changes() -> None:
    engine = DecisionOptimizationEngine(
        pattern_analyzers=(_TestFrictionAnalyzer(),),
        insight_generators=(PatternLinkageInsightGenerator(),),
        recommendation_providers=(),
    )
    context = DecisionOptimizationContext(
        analytics_results=(
            _governance_analytics(allow=9, block=9, require_approval=0),
        ),
    )

    result = engine.run(context, run_at=_stamp())

    assert result.patterns[0].analyzer_id == "test_friction_plugin"
    assert result.patterns[0].summary == "plugin_ok"
    assert result.insights[0].narrative.startswith("Historical analysis")


class _StubRecommendationProvider:
    provider_id = "stub_provider"
    provider_version = "stub-1"

    def recommend(
        self,
        insights: tuple[OptimizationInsight, ...],
        *,
        context: DecisionOptimizationContext,
        generated_at: datetime,
    ) -> tuple[DecisionOptimizationSuggestion, ...]:
        del context
        return tuple(
            DecisionOptimizationSuggestion(
                suggestion_id=f"stub:{item.insight_id}",
                source_decisions=item.source_decision_ids,
                optimization_area=item.optimization_area,
                observation_references=item.data_source_refs,
                rationale="stub rationale",
                confidence=item.confidence,
                generated_at=generated_at,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                insight_ids=(item.insight_id,),
            )
            for item in insights
        )


def test_swappable_recommendation_provider() -> None:
    context = DecisionOptimizationContext(
        analytics_results=(
            _governance_analytics(allow=0, block=4, require_approval=0),
        ),
    )
    shared_generators = (PatternLinkageInsightGenerator(),)
    default_engine = DecisionOptimizationEngine(
        pattern_analyzers=(GovernanceFrictionPatternAnalyzer(),),
        insight_generators=shared_generators,
        recommendation_providers=(HumanReviewRecommendationProvider(),),
    )
    stub_engine = DecisionOptimizationEngine(
        pattern_analyzers=(GovernanceFrictionPatternAnalyzer(),),
        insight_generators=shared_generators,
        recommendation_providers=(_StubRecommendationProvider(),),
    )

    default_result = default_engine.run(context, run_at=_stamp())
    stub_result = stub_engine.run(context, run_at=_stamp())

    assert default_result.patterns == stub_result.patterns
    assert default_result.insights == stub_result.insights
    assert default_result.suggestions[0].provider_id == "human_review"
    assert stub_result.suggestions[0].provider_id == "stub_provider"
    assert stub_result.suggestions[0].rationale == "stub rationale"


def test_empty_context_returns_controlled_result_without_error() -> None:
    engine = default_decision_optimization_engine()
    result = engine.run(
        DecisionOptimizationContext(analytics_results=()),
        run_at=_stamp(),
    )

    assert result.status is OptimizationRunStatus.INSUFFICIENT_DATA
    assert result.patterns == ()
    assert result.insights == ()
    assert result.suggestions == ()
    assert result.aggregate_confidence is None
