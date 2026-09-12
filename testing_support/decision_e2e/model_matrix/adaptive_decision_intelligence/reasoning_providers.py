# © Artur Czarnecki. All rights reserved.

"""Built-in adaptive reasoning providers (extend via new classes)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDataSourceRef,
    AdaptiveDecisionIntelligenceContext,
    AdaptiveReasoningInsight,
    ConfidenceLevel,
    HistoricalEvidence,
)


def _collect_refs(
    evidence: tuple[HistoricalEvidence, ...],
) -> tuple[AdaptiveDataSourceRef, ...]:
    refs: list[AdaptiveDataSourceRef] = []
    for item in evidence:
        refs.extend(item.source_refs)
    return tuple(dict.fromkeys(refs))


def _confidence_from_score(score: float) -> ConfidenceLevel:
    if score >= 0.7:
        return ConfidenceLevel.HIGH
    if score >= 0.4:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


class RiskReasoningProvider:
    provider_id = "risk_reasoning"
    provider_version = "1"

    def reason(
        self,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveReasoningInsight, ...]:
        failed_markers = (
            "failed",
            "manual correction",
            "block",
        )
        hits = [
            item
            for item in context.historical_evidence
            if any(marker in item.summary.lower() for marker in failed_markers)
        ]
        if not hits:
            return ()
        score = min(0.9, 0.35 + 0.15 * len(hits))
        evidence_ids = tuple(item.evidence_id for item in hits)
        return (
            AdaptiveReasoningInsight(
                reasoning_insight_id=f"{self.provider_id}:{context.decision_context_reference.decision_id}",
                reasoning_provider_id=self.provider_id,
                reasoning_provider_version=self.provider_version,
                reasoning_summary=(
                    "Historical evidence indicates elevated risk of friction or "
                    "manual intervention for similar decisions."
                ),
                source_evidence_ids=evidence_ids,
                data_source_refs=_collect_refs(tuple(hits)),
                confidence=_confidence_from_score(score),
                confidence_score=score,
            ),
        )


class QualityReasoningProvider:
    provider_id = "quality_reasoning"
    provider_version = "1"

    def reason(
        self,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveReasoningInsight, ...]:
        capability_hits = [
            item
            for item in context.historical_evidence
            if item.context_provider_id == "capability_context"
        ]
        if not capability_hits:
            return ()
        score = 0.55
        return (
            AdaptiveReasoningInsight(
                reasoning_insight_id=f"{self.provider_id}:{context.decision_context_reference.decision_id}",
                reasoning_provider_id=self.provider_id,
                reasoning_provider_version=self.provider_version,
                reasoning_summary=(
                    "Capability profiles are available — compare model quality "
                    "signals before committing to the same model choice."
                ),
                source_evidence_ids=tuple(item.evidence_id for item in capability_hits),
                data_source_refs=_collect_refs(tuple(capability_hits)),
                confidence=_confidence_from_score(score),
                confidence_score=score,
            ),
        )


class PerformanceReasoningProvider:
    provider_id = "performance_reasoning"
    provider_version = "1"

    def reason(
        self,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveReasoningInsight, ...]:
        analytics_hits = [
            item
            for item in context.historical_evidence
            if item.context_provider_id == "analytics_history"
        ]
        if not analytics_hits:
            return ()
        score = 0.5
        return (
            AdaptiveReasoningInsight(
                reasoning_insight_id=f"{self.provider_id}:{context.decision_context_reference.decision_id}",
                reasoning_provider_id=self.provider_id,
                reasoning_provider_version=self.provider_version,
                reasoning_summary=(
                    "Prior analytics cover related decisions — review throughput "
                    "and outcome patterns when selecting execution paths."
                ),
                source_evidence_ids=tuple(item.evidence_id for item in analytics_hits),
                data_source_refs=_collect_refs(tuple(analytics_hits)),
                confidence=_confidence_from_score(score),
                confidence_score=score,
            ),
        )


def default_reasoning_providers() -> tuple[
    RiskReasoningProvider,
    QualityReasoningProvider,
    PerformanceReasoningProvider,
]:
    return (
        RiskReasoningProvider(),
        QualityReasoningProvider(),
        PerformanceReasoningProvider(),
    )


__all__ = [
    "PerformanceReasoningProvider",
    "QualityReasoningProvider",
    "RiskReasoningProvider",
    "default_reasoning_providers",
]
