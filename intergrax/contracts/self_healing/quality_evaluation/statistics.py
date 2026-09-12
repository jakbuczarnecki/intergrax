# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pure history aggregation for strategy quality evaluation (SELF-HEALING R5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.performance_memory.record import (
    SelfHealingStrategyExecutionOutcome,
    SelfHealingStrategyPerformanceExperience,
)
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment


@dataclass(frozen=True, slots=True)
class StrategyQualityHistoryStatistics:
    execution_count: int
    successful_executions: int
    failed_executions: int
    rolled_back_executions: int
    inconclusive_executions: int
    success_ratio: float
    average_recovery_time_seconds: float | None
    min_recovery_time_seconds: float | None
    max_recovery_time_seconds: float | None
    evidence_refs: tuple[str, ...]


def summarize_strategy_performance_experiences(
    experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
) -> StrategyQualityHistoryStatistics:
    execution_count = len(experiences)
    if execution_count == 0:
        return StrategyQualityHistoryStatistics(
            execution_count=0,
            successful_executions=0,
            failed_executions=0,
            rolled_back_executions=0,
            inconclusive_executions=0,
            success_ratio=0.0,
            average_recovery_time_seconds=None,
            min_recovery_time_seconds=None,
            max_recovery_time_seconds=None,
            evidence_refs=(),
        )

    successful = 0
    failed = 0
    rolled_back = 0
    inconclusive = 0
    recovery_times: list[float] = []
    evidence: list[str] = []
    seen_evidence: set[str] = set()

    for experience in experiences:
        match experience.execution_outcome:
            case SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED:
                successful += 1
            case SelfHealingStrategyExecutionOutcome.REPAIR_FAILED:
                failed += 1
            case SelfHealingStrategyExecutionOutcome.ROLLED_BACK:
                rolled_back += 1
            case SelfHealingStrategyExecutionOutcome.INCONCLUSIVE:
                inconclusive += 1
        recovery_times.append(experience.recovery_time_seconds)
        for ref in experience.evidence_refs:
            if ref not in seen_evidence:
                seen_evidence.add(ref)
                evidence.append(ref)

    success_ratio = successful / execution_count
    return StrategyQualityHistoryStatistics(
        execution_count=execution_count,
        successful_executions=successful,
        failed_executions=failed,
        rolled_back_executions=rolled_back,
        inconclusive_executions=inconclusive,
        success_ratio=success_ratio,
        average_recovery_time_seconds=sum(recovery_times) / execution_count,
        min_recovery_time_seconds=min(recovery_times),
        max_recovery_time_seconds=max(recovery_times),
        evidence_refs=tuple(evidence),
    )


def build_strategy_quality_assessment(
    *,
    tenant_id: str,
    strategy_id: str,
    statistics: StrategyQualityHistoryStatistics,
    quality_score: float,
    evaluator_id: str,
) -> StrategyQualityAssessment:
    return StrategyQualityAssessment(
        tenant_id=tenant_id,
        strategy_id=strategy_id,
        execution_count=statistics.execution_count,
        successful_executions=statistics.successful_executions,
        failed_executions=statistics.failed_executions,
        rolled_back_executions=statistics.rolled_back_executions,
        inconclusive_executions=statistics.inconclusive_executions,
        success_ratio=statistics.success_ratio,
        average_recovery_time_seconds=statistics.average_recovery_time_seconds,
        min_recovery_time_seconds=statistics.min_recovery_time_seconds,
        max_recovery_time_seconds=statistics.max_recovery_time_seconds,
        quality_score=quality_score,
        evaluator_id=evaluator_id,
        evidence_refs=statistics.evidence_refs,
    )


__all__ = [
    "StrategyQualityHistoryStatistics",
    "build_strategy_quality_assessment",
    "summarize_strategy_performance_experiences",
]
