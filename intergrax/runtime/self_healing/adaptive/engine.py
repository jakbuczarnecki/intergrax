# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive self-healing recommendation engine (SELF-HEALING R4)."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from intergrax.contracts.self_healing.adaptive.context import AdaptiveHealingContext
from intergrax.contracts.self_healing.adaptive.learning import AdaptiveHealingLearningRepository
from intergrax.contracts.self_healing.adaptive.recommendation import (
    AdaptiveHealingRecommendation,
    AdaptiveRecommendationStatus,
)
from intergrax.contracts.self_healing.adaptive.registry import AdaptiveHealingPluginDescriptor
from intergrax.contracts.self_healing.adaptive.score import AdaptiveStrategyScore
from intergrax.contracts.self_healing.adaptive.spi import (
    SelfHealingConfidenceEvaluator,
    SelfHealingStrategyRankingProvider,
)
from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy
from intergrax.runtime.self_healing.adaptive.confidence_evaluator import AdaptiveConfidenceEvaluator
from intergrax.runtime.self_healing.adaptive.registries import (
    InMemorySelfHealingConfidenceEvaluatorRegistry,
    InMemorySelfHealingStrategyRankingRegistry,
)


def _merge_scores(
    batches: tuple[tuple[AdaptiveStrategyScore, ...], ...],
) -> tuple[AdaptiveStrategyScore, ...]:
    by_id: dict[str, list[AdaptiveStrategyScore]] = {}
    for batch in batches:
        for score in batch:
            by_id.setdefault(score.strategy_id, []).append(score)
    merged: list[AdaptiveStrategyScore] = []
    for strategy_id, group in by_id.items():
        avg_score = sum(s.calculated_score for s in group) / len(group)
        avg_conf = sum(s.confidence for s in group) / len(group)
        evidence = tuple(dict.fromkeys(ref for s in group for ref in s.evidence_refs))
        factors = group[0].scoring_factors if group else ()
        merged.append(
            AdaptiveStrategyScore(
                strategy_id=strategy_id,
                calculated_score=avg_score,
                confidence=avg_conf if evidence else 0.0,
                evidence_refs=evidence,
                scoring_factors=factors,
            ),
        )
    merged.sort(key=lambda s: s.calculated_score, reverse=True)
    return tuple(merged)


def _invoke_ranking(
    provider: SelfHealingStrategyRankingProvider,
    context: AdaptiveHealingContext,
) -> tuple[AdaptiveStrategyScore, ...]:
    return provider.rank_strategies(context)


class AdaptiveSelfHealingEngine:
    """
    Collects adaptive context, runs ranking/confidence plugins, returns recommendation only.
    """

    def __init__(
        self,
        ranking_registry: InMemorySelfHealingStrategyRankingRegistry,
        confidence_registry: InMemorySelfHealingConfidenceEvaluatorRegistry | None = None,
        *,
        learning_repository: AdaptiveHealingLearningRepository | None = None,
        default_confidence_evaluator: AdaptiveConfidenceEvaluator | None = None,
    ) -> None:
        self._ranking_registry = ranking_registry
        self._confidence_registry = confidence_registry or InMemorySelfHealingConfidenceEvaluatorRegistry()
        self._learning = learning_repository
        self._default_confidence = default_confidence_evaluator or AdaptiveConfidenceEvaluator()

    def build_context(
        self,
        healing_context: SelfHealingContext,
        strategy_candidates: tuple[SelfHealingStrategy, ...],
        *,
        execution_context_ref: str | None = None,
    ) -> AdaptiveHealingContext:
        tenant_id = healing_context.tenant_id
        validation = ()
        rollback = ()
        similarity = ()
        if self._learning is not None:
            validation = self._learning.validation_quality_for_tenant(tenant_id)
            rollback = self._learning.rollback_history_for_tenant(tenant_id)
            similarity = self._learning.context_similarity_refs_for_tenant(tenant_id)
        evidence = tuple(
            dict.fromkeys(
                list(healing_context.diagnostic_investigation.evidence_refs)
                + list(similarity),
            ),
        )
        return AdaptiveHealingContext(
            tenant_id=tenant_id,
            strategy_candidates=strategy_candidates,
            historical_outcomes=healing_context.historical_outcomes,
            evidence_refs=evidence,
            execution_context_ref=execution_context_ref,
            validation_quality=validation,
            rollback_history=rollback,
            context_similarity_refs=similarity,
        )

    def recommend(
        self,
        context: AdaptiveHealingContext,
    ) -> AdaptiveHealingRecommendation:
        ranking_entries = self._ranking_registry.list_for_tenant(context.tenant_id)
        insights: list[str] = []
        status = AdaptiveRecommendationStatus.OK
        score_batches: list[tuple[AdaptiveStrategyScore, ...]] = []

        if not ranking_entries:
            status = AdaptiveRecommendationStatus.PLUGIN_UNAVAILABLE
            insights.append("no ranking providers registered")
        else:
            degraded = False
            unavailable = False
            with ThreadPoolExecutor(max_workers=min(4, len(ranking_entries))) as pool:
                futures: list[tuple[Future[tuple[AdaptiveStrategyScore, ...]], AdaptiveHealingPluginDescriptor]] = []
                for provider, descriptor in ranking_entries:
                    futures.append((pool.submit(_invoke_ranking, provider, context), descriptor))
                for future, descriptor in futures:
                    try:
                        batch = future.result(timeout=descriptor.timeout_seconds)
                    except FuturesTimeoutError:
                        unavailable = True
                        insights.append(f"ranking provider {descriptor.plugin_id} timed out")
                        continue
                    except Exception as exc:  # noqa: BLE001 — plugin isolation
                        degraded = True
                        insights.append(f"ranking provider {descriptor.plugin_id} failed: {exc}")
                        continue
                    if not batch:
                        degraded = True
                        insights.append(f"ranking provider {descriptor.plugin_id} returned empty scores")
                        continue
                    score_batches.append(batch)
            if unavailable and not score_batches:
                status = AdaptiveRecommendationStatus.PLUGIN_UNAVAILABLE
            elif degraded or unavailable:
                status = AdaptiveRecommendationStatus.DEGRADED_ADAPTIVE_INTELLIGENCE

        merged_scores = _merge_scores(tuple(score_batches)) if score_batches else ()

        confidence_entries = self._confidence_registry.list_for_tenant(context.tenant_id)
        confidence = 0.0
        confidence_evidence: tuple[str, ...] = ()
        confidence_explanation = ""
        evaluator: SelfHealingConfidenceEvaluator = self._default_confidence
        if confidence_entries:
            evaluator = confidence_entries[0][0]
        try:
            confidence, confidence_evidence, confidence_explanation = evaluator.evaluate_confidence(
                context,
                merged_scores,
            )
        except Exception as exc:  # noqa: BLE001 — plugin isolation
            status = AdaptiveRecommendationStatus.DEGRADED_ADAPTIVE_INTELLIGENCE
            insights.append(f"confidence evaluator failed: {exc}")
            confidence = 0.0
            confidence_evidence = tuple(context.evidence_refs)
            confidence_explanation = "confidence evaluation degraded"

        if status == AdaptiveRecommendationStatus.OK and not confidence_evidence:
            confidence = 0.0
            status = AdaptiveRecommendationStatus.DEGRADED_ADAPTIVE_INTELLIGENCE
            confidence_explanation = "confidence requires evidence"

        order = tuple(s.strategy_id for s in merged_scores)
        if not order:
            order = tuple(s.strategy_id for s in context.strategy_candidates)

        return AdaptiveHealingRecommendation(
            tenant_id=context.tenant_id,
            recommended_strategy_order=order,
            strategy_scores=merged_scores,
            overall_confidence=confidence,
            status=status,
            evidence_refs=confidence_evidence if confidence_evidence else tuple(context.evidence_refs),
            adaptive_insights=tuple(insights),
            confidence_explanation=confidence_explanation,
        )


__all__ = ["AdaptiveSelfHealingEngine"]
