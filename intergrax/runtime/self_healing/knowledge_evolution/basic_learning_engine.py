# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Statistical strategy learning engine (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.self_healing.knowledge_evolution.comparison import StrategyComparisonResult
from intergrax.contracts.self_healing.knowledge_evolution.confidence import StrategyKnowledgeConfidenceLevel
from intergrax.contracts.self_healing.knowledge_evolution.evolution import (
    StrategyKnowledgeEvolutionContext,
    StrategyKnowledgeEvolutionRevisionMetadata,
    StrategyKnowledgeEvolutionResult,
)
from intergrax.contracts.self_healing.knowledge_evolution.metrics import StrategyMetricBundle
from intergrax.contracts.self_healing.knowledge_evolution.profile import (
    StrategyKnowledgeFreshness,
    StrategyKnowledgeObservationSummary,
    StrategyKnowledgeProfile,
    StrategyKnowledgeQualitySnapshot,
    StrategyKnowledgeRevision,
    mint_strategy_knowledge_profile_id,
    mint_strategy_knowledge_revision_id,
)
from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience
from intergrax.contracts.self_healing.quality_evaluation.statistics import summarize_strategy_performance_experiences
from intergrax.runtime.self_healing.knowledge_evolution.fingerprint import build_experience_set_fingerprint

_SAMPLE_LIMIT = 20


def _confidence_for_count(execution_count: int) -> StrategyKnowledgeConfidenceLevel:
    if execution_count == 0:
        return StrategyKnowledgeConfidenceLevel.INSUFFICIENT_DATA
    if execution_count < 5:
        return StrategyKnowledgeConfidenceLevel.LOW
    if execution_count < 20:
        return StrategyKnowledgeConfidenceLevel.MEDIUM
    return StrategyKnowledgeConfidenceLevel.HIGH


@dataclass(frozen=True, slots=True)
class BasicStrategyLearningEngine:
    engine_id: str = "platform.basic_strategy_learning"

    def evolve(
        self,
        context: StrategyKnowledgeEvolutionContext,
        current_profile: StrategyKnowledgeProfile | None,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        metrics: StrategyMetricBundle,
        comparison: StrategyComparisonResult | None,
    ) -> StrategyKnowledgeEvolutionResult:
        knowledge_context = context.knowledge_context
        experience_fingerprint = build_experience_set_fingerprint(experiences)
        if current_profile is not None and current_profile.input_experience_fingerprint == experience_fingerprint:
            return StrategyKnowledgeEvolutionResult(
                proposed_profile=None,
                revision_metadata=None,
                no_change=True,
            )
        if not experiences:
            return StrategyKnowledgeEvolutionResult(
                proposed_profile=None,
                revision_metadata=None,
                no_change=True,
            )

        statistics = summarize_strategy_performance_experiences(experiences)
        ordered = sorted(experiences, key=lambda row: row.recorded_at)
        earliest = ordered[0].recorded_at
        latest = ordered[-1].recorded_at
        experience_ids = tuple(row.experience_id for row in ordered)
        observation = StrategyKnowledgeObservationSummary(
            experience_count=statistics.execution_count,
            experience_id_sample=experience_ids[-_SAMPLE_LIMIT:],
            earliest_recorded_at=earliest,
            latest_recorded_at=latest,
            evidence_refs=statistics.evidence_refs,
        )
        assessment = context.optional_quality_assessment
        quality_snapshot: StrategyKnowledgeQualitySnapshot | None
        if assessment is not None:
            quality_snapshot = StrategyKnowledgeQualitySnapshot(
                execution_count=assessment.execution_count,
                success_ratio=assessment.success_ratio,
                average_recovery_time_seconds=assessment.average_recovery_time_seconds,
                quality_score=assessment.quality_score,
                evaluator_id=assessment.evaluator_id,
                evidence_refs=assessment.evidence_refs,
            )
            quality_score = assessment.quality_score
        else:
            quality_score = statistics.success_ratio if statistics.execution_count > 0 else 0.0
            quality_snapshot = StrategyKnowledgeQualitySnapshot(
                execution_count=statistics.execution_count,
                success_ratio=statistics.success_ratio,
                average_recovery_time_seconds=statistics.average_recovery_time_seconds,
                quality_score=quality_score,
                evaluator_id=self.engine_id,
                evidence_refs=statistics.evidence_refs,
            )

        next_version = 1 if current_profile is None else current_profile.knowledge_version + 1
        previous_version = current_profile.knowledge_version if current_profile is not None else None
        profile_id = (
            current_profile.profile_id if current_profile is not None else mint_strategy_knowledge_profile_id()
        )
        derived_at = datetime.now(tz=timezone.utc)
        staleness_policy_id = context.freshness_policy_id
        profile = StrategyKnowledgeProfile(
            profile_id=profile_id,
            tenant_id=knowledge_context.tenant_id,
            strategy_id=knowledge_context.strategy_id,
            context_fingerprint=knowledge_context.context_fingerprint,
            context_refs=knowledge_context.context_refs,
            observation_summary=observation,
            quality_snapshot=quality_snapshot,
            confidence_label=_confidence_for_count(statistics.execution_count),
            freshness=StrategyKnowledgeFreshness(
                last_evidence_at=latest,
                staleness_policy_id=staleness_policy_id,
                ttl_hint_seconds=None,
            ),
            knowledge_version=next_version,
            supersedes_version=previous_version,
            derived_at=derived_at,
            learning_engine_id=self.engine_id,
            input_experience_fingerprint=experience_fingerprint,
            operating_context=context.resolved_operating_context,
        )
        metric_refs = tuple(
            f"metric://{metrics.provider_id}/{metric.name}" for metric in metrics.metrics
        )
        comparison_policy_id = comparison.policy_id if comparison is not None else None
        assessment_refs = assessment.evidence_refs if assessment is not None else ()
        change_summary = (
            f"Knowledge version {next_version}: "
            f"{statistics.execution_count} experiences for {knowledge_context.strategy_id} "
            f"in context {knowledge_context.context_fingerprint}."
        )
        if comparison is not None:
            change_summary = f"{change_summary} Comparison: {comparison.rationale}"
        metadata = StrategyKnowledgeEvolutionRevisionMetadata(
            change_summary=change_summary,
            input_experience_ids=experience_ids,
            input_assessment_refs=assessment_refs,
            metric_snapshot_refs=metric_refs,
            comparison_policy_id=comparison_policy_id,
            previous_knowledge_version=previous_version,
        )
        revision = StrategyKnowledgeRevision(
            revision_id=mint_strategy_knowledge_revision_id(),
            profile=profile,
            change_summary=change_summary,
            trigger=context.trigger,
            trigger_refs=context.trigger_refs,
            input_experience_ids=experience_ids,
            input_assessment_refs=assessment_refs,
            metric_snapshot_refs=metric_refs,
            comparison_policy_id=comparison_policy_id,
            previous_knowledge_version=previous_version,
            evolution_mechanism_id=self.engine_id,
            recorded_at=derived_at,
        )
        return StrategyKnowledgeEvolutionResult(
            proposed_profile=profile,
            revision_metadata=metadata,
            no_change=False,
            proposed_revision=revision,
        )


__all__ = ["BasicStrategyLearningEngine"]
