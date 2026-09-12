# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy knowledge evolution orchestration (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from intergrax.contracts.self_healing.knowledge_evolution.comparison import (
    StrategyComparisonPolicy,
    StrategyComparisonResult,
    StrategyComparisonScope,
    StrategyComparisonSubject,
)
from intergrax.contracts.self_healing.knowledge_evolution.contextual.context_provider import (
    StrategyContextProvider,
    StrategyContextResolutionRequest,
)
from intergrax.contracts.self_healing.knowledge_evolution.contextual.freshness_policy import (
    KnowledgeFreshnessPolicy,
)
from intergrax.contracts.self_healing.knowledge_evolution.contextual.operating_context import (
    merge_operating_contexts,
)
from intergrax.contracts.self_healing.knowledge_evolution.engine import StrategyLearningEngine
from intergrax.contracts.self_healing.knowledge_evolution.evolution import (
    StrategyKnowledgeEvolutionContext,
    StrategyKnowledgeEvolutionResult,
)
from intergrax.contracts.self_healing.knowledge_evolution.metrics import StrategyMetricProvider, StrategyMetricScope
from intergrax.contracts.self_healing.knowledge_evolution.query import (
    StrategyKnowledgeProfileQuery,
    StrategyKnowledgeRevisionQuery,
)
from intergrax.contracts.self_healing.knowledge_evolution.repository import StrategyKnowledgeRepository
from intergrax.contracts.self_healing.performance_memory.query import StrategyPerformanceMemoryQuery
from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience
from intergrax.contracts.self_healing.performance_memory.repository import StrategyPerformanceMemoryRepository
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria
from intergrax.runtime.self_healing.knowledge_evolution.governance.service import StrategyKnowledgeGovernanceService
from intergrax.runtime.self_healing.quality_evaluation.service import StrategyQualityEvaluationService


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeEvolutionService:
    performance_memory: StrategyPerformanceMemoryRepository
    knowledge_repository: StrategyKnowledgeRepository
    learning_engine: StrategyLearningEngine
    metric_provider: StrategyMetricProvider
    comparison_policy: StrategyComparisonPolicy | None = None
    quality_evaluation: StrategyQualityEvaluationService | None = None
    context_providers: tuple[StrategyContextProvider, ...] = field(default_factory=tuple)
    freshness_policy: KnowledgeFreshnessPolicy | None = None
    knowledge_governance: StrategyKnowledgeGovernanceService | None = None

    def evolve(self, context: StrategyKnowledgeEvolutionContext) -> StrategyKnowledgeEvolutionResult:
        context = self._enrich_context(context)
        knowledge_context = context.knowledge_context
        if self._revision_exists_for_trigger(context):
            return StrategyKnowledgeEvolutionResult(
                proposed_profile=None,
                revision_metadata=None,
                no_change=True,
            )
        experiences = self.performance_memory.query(
            StrategyPerformanceMemoryQuery(
                tenant_id=knowledge_context.tenant_id,
                strategy_id=knowledge_context.strategy_id,
            ),
        )
        if knowledge_context.time_horizon_experience_limit is not None:
            experiences = experiences[-knowledge_context.time_horizon_experience_limit :]
        reference_time = datetime.now(tz=timezone.utc)
        if self.freshness_policy is not None:
            experiences = self.freshness_policy.order_experiences(experiences, reference_time)

        profile_query = StrategyKnowledgeProfileQuery(
            tenant_id=knowledge_context.tenant_id,
            strategy_id=knowledge_context.strategy_id,
            context_fingerprint=knowledge_context.context_fingerprint,
        )
        current_profile = self.knowledge_repository.get_latest_profile(profile_query)

        metric_scope = StrategyMetricScope(
            tenant_id=knowledge_context.tenant_id,
            strategy_id=knowledge_context.strategy_id,
            context_fingerprint=knowledge_context.context_fingerprint,
        )
        metrics = self.metric_provider.collect(metric_scope, experiences)

        assessment = context.optional_quality_assessment
        if assessment is None and self.quality_evaluation is not None:
            assessment = self.quality_evaluation.assess(
                StrategyQualityEvaluationCriteria(
                    tenant_id=knowledge_context.tenant_id,
                    strategy_id=knowledge_context.strategy_id,
                ),
            )

        comparison = self._maybe_compare(context, experiences, assessment)
        result = self.learning_engine.evolve(
            context=context,
            current_profile=current_profile,
            experiences=experiences,
            metrics=metrics,
            comparison=comparison,
        )
        if result.no_change:
            return result
        if result.proposed_revision is None:
            raise ValueError("learning engine must supply proposed_revision when knowledge changes")
        self.knowledge_repository.append_revision(result.proposed_revision)
        if self.knowledge_governance is not None:
            self.knowledge_governance.record_knowledge_evolution(
                result.proposed_revision,
                knowledge_repository=self.knowledge_repository,
            )
        return result

    def _revision_exists_for_trigger(self, context: StrategyKnowledgeEvolutionContext) -> bool:
        knowledge_context = context.knowledge_context
        revisions = self.knowledge_repository.list_revisions(
            StrategyKnowledgeRevisionQuery(
                tenant_id=knowledge_context.tenant_id,
                strategy_id=knowledge_context.strategy_id,
                context_fingerprint=knowledge_context.context_fingerprint,
            ),
        )
        for revision in revisions:
            if revision.trigger == context.trigger and revision.trigger_refs == context.trigger_refs:
                return True
        return False

    def _maybe_compare(
        self,
        context: StrategyKnowledgeEvolutionContext,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        assessment: StrategyQualityAssessment | None,
    ) -> StrategyComparisonResult | None:
        if self.comparison_policy is None or len(context.comparison_strategy_ids) < 2:
            return None

        left_id = context.comparison_strategy_ids[0]
        right_id = context.comparison_strategy_ids[1]
        left_experiences = tuple(row for row in experiences if row.strategy_id == left_id)
        right_experiences = tuple(row for row in experiences if row.strategy_id == right_id)
        left_metrics = self.metric_provider.collect(
            StrategyMetricScope(
                tenant_id=context.knowledge_context.tenant_id,
                strategy_id=left_id,
                context_fingerprint=context.knowledge_context.context_fingerprint,
            ),
            left_experiences,  # type: ignore[arg-type]
        )
        right_metrics = self.metric_provider.collect(
            StrategyMetricScope(
                tenant_id=context.knowledge_context.tenant_id,
                strategy_id=right_id,
                context_fingerprint=context.knowledge_context.context_fingerprint,
            ),
            right_experiences,
        )
        left_assessment = assessment
        right_assessment = None
        if self.quality_evaluation is not None:
            right_assessment = self.quality_evaluation.assess(
                StrategyQualityEvaluationCriteria(
                    tenant_id=context.knowledge_context.tenant_id,
                    strategy_id=right_id,
                ),
            )
        reference_time = datetime.now(tz=timezone.utc)
        operating_context = context.resolved_operating_context
        return self.comparison_policy.compare(
            StrategyComparisonScope(
                tenant_id=context.knowledge_context.tenant_id,
                context_fingerprint=context.knowledge_context.context_fingerprint,
                dimension_weights_ref=None,
                operating_context=operating_context,
            ),
            StrategyComparisonSubject(
                strategy_id=left_id,
                metric_bundle=left_metrics,
                quality_assessment=left_assessment if left_id == context.knowledge_context.strategy_id else None,
                operating_context=operating_context,
                knowledge_freshness_score=self._aggregate_freshness(left_experiences, reference_time),
            ),
            StrategyComparisonSubject(
                strategy_id=right_id,
                metric_bundle=right_metrics,
                quality_assessment=right_assessment,
                knowledge_freshness_score=self._aggregate_freshness(right_experiences, reference_time),
            ),
        )

    def _enrich_context(self, context: StrategyKnowledgeEvolutionContext) -> StrategyKnowledgeEvolutionContext:
        operating_context = context.resolved_operating_context
        if operating_context is None and self.context_providers:
            request = StrategyContextResolutionRequest(
                tenant_id=context.knowledge_context.tenant_id,
                strategy_id=context.knowledge_context.strategy_id,
                evolution_scope=context.knowledge_context,
                trigger_refs=context.trigger_refs,
            )
            resolved = tuple(
                row
                for provider in self.context_providers
                for row in (provider.resolve(request),)
                if row is not None
            )
            operating_context = merge_operating_contexts(resolved)
        freshness_policy_id = context.freshness_policy_id
        if freshness_policy_id is None and self.freshness_policy is not None:
            freshness_policy_id = self.freshness_policy.policy_id
        if operating_context == context.resolved_operating_context and freshness_policy_id == context.freshness_policy_id:
            return context
        return StrategyKnowledgeEvolutionContext(
            knowledge_context=context.knowledge_context,
            trigger=context.trigger,
            trigger_refs=context.trigger_refs,
            optional_quality_assessment=context.optional_quality_assessment,
            comparison_strategy_ids=context.comparison_strategy_ids,
            resolved_operating_context=operating_context,
            freshness_policy_id=freshness_policy_id,
        )

    def _aggregate_freshness(
        self,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        reference_time: datetime,
    ) -> float | None:
        if self.freshness_policy is None or not experiences:
            return None
        scores = [
            self.freshness_policy.freshness_score(row.recorded_at, reference_time) for row in experiences
        ]
        return sum(scores) / len(scores)


__all__ = ["StrategyKnowledgeEvolutionService"]
