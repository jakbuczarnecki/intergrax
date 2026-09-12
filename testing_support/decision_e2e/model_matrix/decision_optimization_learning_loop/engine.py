# © Artur Czarnecki. All rights reserved.

"""Decision optimization orchestration via injected plugins (DS-E2E-15J-L9)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    OPTIMIZATION_TASK_ID,
    OPTIMIZATION_VERSION,
    ConfidenceLevel,
    DecisionOptimizationAuditMetadata,
    DecisionOptimizationContext,
    DecisionOptimizationResult,
    DecisionOptimizationSuggestion,
    OptimizationRunStatus,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.insight_generators import (
    default_insight_generators,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.pattern_analyzers import (
    default_pattern_analyzers,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.protocol import (
    OptimizationInsightGenerator,
    OptimizationPatternAnalyzer,
    OptimizationRecommendationProvider,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.recommendation_providers import (
    default_recommendation_providers,
)

_CONFIDENCE_ORDER = (
    ConfidenceLevel.LOW,
    ConfidenceLevel.MEDIUM,
    ConfidenceLevel.HIGH,
)


def _aggregate_confidence(
    suggestions: tuple[DecisionOptimizationSuggestion, ...],
) -> ConfidenceLevel | None:
    if not suggestions:
        return None
    best = ConfidenceLevel.LOW
    for item in suggestions:
        if _CONFIDENCE_ORDER.index(item.confidence) > _CONFIDENCE_ORDER.index(best):
            best = item.confidence
    return best


def _has_input_data(context: DecisionOptimizationContext) -> bool:
    return bool(
        context.analytics_results
        or context.observations
        or context.lifecycle_records
        or context.capability_profiles
        or context.governance_decisions
    )


def _analytics_refs(context: DecisionOptimizationContext) -> tuple[str, ...]:
    return tuple(
        f"{item.audit.analyzer_id}:{item.payload_kind.value}"
        for item in context.analytics_results
    )


@dataclass(frozen=True, slots=True)
class DecisionOptimizationEngine:
    pattern_analyzers: tuple[OptimizationPatternAnalyzer, ...]
    insight_generators: tuple[OptimizationInsightGenerator, ...]
    recommendation_providers: tuple[OptimizationRecommendationProvider, ...]

    def run(
        self,
        context: DecisionOptimizationContext,
        *,
        run_at: datetime | None = None,
    ) -> DecisionOptimizationResult:
        stamp = run_at or datetime.now(tz=UTC)
        if not _has_input_data(context):
            return DecisionOptimizationResult(
                optimization_task_id=OPTIMIZATION_TASK_ID,
                status=OptimizationRunStatus.INSUFFICIENT_DATA,
                audit=DecisionOptimizationAuditMetadata(
                    engine_task_id=OPTIMIZATION_TASK_ID,
                    engine_version=OPTIMIZATION_VERSION,
                    pattern_analyzer_ids=tuple(
                        item.analyzer_id for item in self.pattern_analyzers
                    ),
                    insight_generator_ids=tuple(
                        item.generator_id for item in self.insight_generators
                    ),
                    recommendation_provider_ids=tuple(
                        item.provider_id for item in self.recommendation_providers
                    ),
                    analyzed_at=stamp,
                    analytics_result_refs=(),
                    observation_decision_ids=(),
                ),
                patterns=(),
                insights=(),
                suggestions=(),
                aggregate_confidence=None,
            )

        patterns = tuple(
            pattern
            for analyzer in self.pattern_analyzers
            for pattern in analyzer.detect_patterns(context)
        )
        insights = tuple(
            insight
            for generator in self.insight_generators
            for insight in generator.generate_insights(patterns, context=context)
        )
        suggestions = tuple(
            suggestion
            for provider in self.recommendation_providers
            for suggestion in provider.recommend(
                insights,
                context=context,
                generated_at=stamp,
            )
        )
        observation_ids = tuple(
            dict.fromkeys(item.decision_id for item in context.observations)
        )
        status = (
            OptimizationRunStatus.COMPLETE
            if patterns or insights or suggestions
            else OptimizationRunStatus.INSUFFICIENT_DATA
        )
        return DecisionOptimizationResult(
            optimization_task_id=OPTIMIZATION_TASK_ID,
            status=status,
            audit=DecisionOptimizationAuditMetadata(
                engine_task_id=OPTIMIZATION_TASK_ID,
                engine_version=OPTIMIZATION_VERSION,
                pattern_analyzer_ids=tuple(
                    item.analyzer_id for item in self.pattern_analyzers
                ),
                insight_generator_ids=tuple(
                    item.generator_id for item in self.insight_generators
                ),
                recommendation_provider_ids=tuple(
                    item.provider_id for item in self.recommendation_providers
                ),
                analyzed_at=stamp,
                analytics_result_refs=_analytics_refs(context),
                observation_decision_ids=observation_ids,
            ),
            patterns=patterns,
            insights=insights,
            suggestions=suggestions,
            aggregate_confidence=_aggregate_confidence(suggestions),
        )


def default_decision_optimization_engine(
    *,
    pattern_analyzers: tuple[OptimizationPatternAnalyzer, ...] | None = None,
    insight_generators: tuple[OptimizationInsightGenerator, ...] | None = None,
    recommendation_providers: tuple[OptimizationRecommendationProvider, ...]
    | None = None,
) -> DecisionOptimizationEngine:
    return DecisionOptimizationEngine(
        pattern_analyzers=pattern_analyzers or default_pattern_analyzers(),
        insight_generators=insight_generators or default_insight_generators(),
        recommendation_providers=recommendation_providers
        or default_recommendation_providers(),
    )


__all__ = [
    "DecisionOptimizationEngine",
    "default_decision_optimization_engine",
]
