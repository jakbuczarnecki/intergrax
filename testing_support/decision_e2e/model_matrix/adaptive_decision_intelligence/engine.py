# © Artur Czarnecki. All rights reserved.

"""Adaptive decision intelligence orchestration via injected plugins (DS-E2E-15J-L10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    ADAPTIVE_INTELLIGENCE_TASK_ID,
    ADAPTIVE_INTELLIGENCE_VERSION,
    AdaptiveDataSourceRef,
    AdaptiveDecisionInsight,
    AdaptiveDecisionIntelligenceAuditMetadata,
    AdaptiveDecisionIntelligenceContext,
    AdaptiveDecisionIntelligenceInput,
    AdaptiveDecisionIntelligenceResult,
    AdaptiveDecisionRecommendation,
    AdaptiveIntelligenceRunStatus,
    AdaptiveReasoningInsight,
    HistoricalEvidence,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.context_providers import (
    default_context_providers,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.protocol import (
    AdaptiveDecisionContextProvider,
    AdaptiveReasoningProvider,
    AdaptiveRecommendationProvider,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.reasoning_providers import (
    default_reasoning_providers,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.recommendation_providers import (
    default_recommendation_providers,
)


def _merge_evidence(
    providers: tuple[AdaptiveDecisionContextProvider, ...],
    input_data: AdaptiveDecisionIntelligenceInput,
) -> tuple[HistoricalEvidence, ...]:
    merged: list[HistoricalEvidence] = []
    for provider in providers:
        merged.extend(provider.contribute(input_data))
    return tuple(merged)


def _data_source_refs(
    evidence: tuple[HistoricalEvidence, ...],
) -> tuple[AdaptiveDataSourceRef, ...]:
    refs: list[AdaptiveDataSourceRef] = []
    for item in evidence:
        refs.extend(item.source_refs)
    return tuple(dict.fromkeys(refs))


def _build_context(
    providers: tuple[AdaptiveDecisionContextProvider, ...],
    input_data: AdaptiveDecisionIntelligenceInput,
) -> AdaptiveDecisionIntelligenceContext:
    evidence = _merge_evidence(providers, input_data)
    return AdaptiveDecisionIntelligenceContext(
        decision_context_reference=input_data.decision_context_reference,
        historical_evidence=evidence,
        data_source_refs=_data_source_refs(evidence),
        context_provider_ids=tuple(item.provider_id for item in providers),
        context_provider_versions=tuple(item.provider_version for item in providers),
    )


def _has_input_data(input_data: AdaptiveDecisionIntelligenceInput) -> bool:
    opt = input_data.optimization_result
    has_opt = opt is not None and (opt.suggestions or opt.insights or opt.patterns)
    return bool(
        input_data.lifecycle_records
        or input_data.analytics_results
        or has_opt
        or input_data.capability_profiles
        or input_data.governance_decisions
    )


def _assemble_insights(
    *,
    context: AdaptiveDecisionIntelligenceContext,
    reasoning_insights: tuple[AdaptiveReasoningInsight, ...],
    recommendations: tuple[AdaptiveDecisionRecommendation, ...],
    generated_at: datetime,
    reasoning_provider_ids: tuple[str, ...],
) -> tuple[AdaptiveDecisionInsight, ...]:
    if not recommendations:
        return ()
    by_reasoning: dict[str, AdaptiveReasoningInsight] = {
        item.reasoning_insight_id: item for item in reasoning_insights
    }
    insights: list[AdaptiveDecisionInsight] = []
    for recommendation in recommendations:
        linked = [
            by_reasoning[item_id]
            for item_id in recommendation.linked_reasoning_insight_ids
            if item_id in by_reasoning
        ]
        reasoning_summary = (
            linked[0].reasoning_summary if linked else "No linked reasoning insight."
        )
        evidence_ids = {
            eid for insight in linked for eid in insight.source_evidence_ids
        }
        historical = tuple(
            item
            for item in context.historical_evidence
            if item.evidence_id in evidence_ids
        )
        if not historical:
            historical = context.historical_evidence
        data_refs = _data_source_refs(historical)
        insights.append(
            AdaptiveDecisionInsight(
                insight_id=f"adi:{recommendation.recommendation_id}",
                decision_context_reference=context.decision_context_reference,
                historical_evidence=historical,
                reasoning_summary=reasoning_summary,
                recommendation=recommendation,
                confidence=recommendation.confidence,
                confidence_score=recommendation.confidence_score,
                generated_at=generated_at,
                context_provider_ids=context.context_provider_ids,
                reasoning_provider_ids=reasoning_provider_ids,
                recommendation_provider_id=recommendation.recommendation_provider_id,
                data_source_refs=data_refs,
            )
        )
    return tuple(insights)


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionIntelligenceEngine:
    context_providers: tuple[AdaptiveDecisionContextProvider, ...]
    reasoning_providers: tuple[AdaptiveReasoningProvider, ...]
    recommendation_providers: tuple[AdaptiveRecommendationProvider, ...]

    def build_context(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> AdaptiveDecisionIntelligenceContext:
        return _build_context(self.context_providers, input_data)

    def assist(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
        *,
        run_at: datetime | None = None,
    ) -> AdaptiveDecisionIntelligenceResult:
        stamp = run_at or datetime.now(tz=UTC)
        ref = input_data.decision_context_reference
        empty_audit = AdaptiveDecisionIntelligenceAuditMetadata(
            engine_task_id=ADAPTIVE_INTELLIGENCE_TASK_ID,
            engine_version=ADAPTIVE_INTELLIGENCE_VERSION,
            context_provider_ids=tuple(
                item.provider_id for item in self.context_providers
            ),
            reasoning_provider_ids=tuple(
                item.provider_id for item in self.reasoning_providers
            ),
            recommendation_provider_ids=tuple(
                item.provider_id for item in self.recommendation_providers
            ),
            analyzed_at=stamp,
            decision_context_reference=ref,
            data_source_refs=(),
        )
        if not _has_input_data(input_data):
            return AdaptiveDecisionIntelligenceResult(
                intelligence_task_id=ADAPTIVE_INTELLIGENCE_TASK_ID,
                status=AdaptiveIntelligenceRunStatus.INSUFFICIENT_CONTEXT,
                audit=empty_audit,
                reasoning_insights=(),
                recommendations=(),
                insights=(),
            )

        context = _build_context(self.context_providers, input_data)
        if not context.historical_evidence:
            return AdaptiveDecisionIntelligenceResult(
                intelligence_task_id=ADAPTIVE_INTELLIGENCE_TASK_ID,
                status=AdaptiveIntelligenceRunStatus.INSUFFICIENT_CONTEXT,
                audit=AdaptiveDecisionIntelligenceAuditMetadata(
                    engine_task_id=empty_audit.engine_task_id,
                    engine_version=empty_audit.engine_version,
                    context_provider_ids=empty_audit.context_provider_ids,
                    reasoning_provider_ids=empty_audit.reasoning_provider_ids,
                    recommendation_provider_ids=empty_audit.recommendation_provider_ids,
                    analyzed_at=stamp,
                    decision_context_reference=ref,
                    data_source_refs=context.data_source_refs,
                ),
                reasoning_insights=(),
                recommendations=(),
                insights=(),
            )

        reasoning_insights = tuple(
            insight
            for provider in self.reasoning_providers
            for insight in provider.reason(context)
        )
        recommendations = tuple(
            recommendation
            for provider in self.recommendation_providers
            for recommendation in provider.recommend(
                reasoning_insights,
                context=context,
            )
        )
        reasoning_ids = tuple(
            dict.fromkeys(item.provider_id for item in self.reasoning_providers)
        )
        insights = _assemble_insights(
            context=context,
            reasoning_insights=reasoning_insights,
            recommendations=recommendations,
            generated_at=stamp,
            reasoning_provider_ids=reasoning_ids,
        )
        status = (
            AdaptiveIntelligenceRunStatus.COMPLETE
            if insights or reasoning_insights or recommendations
            else AdaptiveIntelligenceRunStatus.INSUFFICIENT_CONTEXT
        )
        return AdaptiveDecisionIntelligenceResult(
            intelligence_task_id=ADAPTIVE_INTELLIGENCE_TASK_ID,
            status=status,
            audit=AdaptiveDecisionIntelligenceAuditMetadata(
                engine_task_id=ADAPTIVE_INTELLIGENCE_TASK_ID,
                engine_version=ADAPTIVE_INTELLIGENCE_VERSION,
                context_provider_ids=context.context_provider_ids,
                reasoning_provider_ids=reasoning_ids,
                recommendation_provider_ids=tuple(
                    item.provider_id for item in self.recommendation_providers
                ),
                analyzed_at=stamp,
                decision_context_reference=ref,
                data_source_refs=context.data_source_refs,
            ),
            reasoning_insights=reasoning_insights,
            recommendations=recommendations,
            insights=insights,
        )


def default_adaptive_decision_intelligence_engine(
    *,
    context_providers: tuple[AdaptiveDecisionContextProvider, ...] | None = None,
    reasoning_providers: tuple[AdaptiveReasoningProvider, ...] | None = None,
    recommendation_providers: tuple[AdaptiveRecommendationProvider, ...] | None = None,
) -> AdaptiveDecisionIntelligenceEngine:
    return AdaptiveDecisionIntelligenceEngine(
        context_providers=context_providers or default_context_providers(),
        reasoning_providers=reasoning_providers or default_reasoning_providers(),
        recommendation_providers=recommendation_providers
        or default_recommendation_providers(),
    )


__all__ = [
    "AdaptiveDecisionIntelligenceEngine",
    "default_adaptive_decision_intelligence_engine",
]
