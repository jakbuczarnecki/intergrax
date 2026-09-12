# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy performance projection — single store authority (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.selection.performance import SelfHealingStrategyPerformance
from intergrax.contracts.self_healing.validation.decision import ValidationDecisionStatus
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.runtime.self_healing.outcome_learning import InMemorySelfHealingStrategyQualityStore


@dataclass
class InMemorySelfHealingStrategyPerformanceStore:
    _profiles: dict[tuple[str, str], SelfHealingStrategyPerformance] = field(default_factory=dict)

    def get(self, *, strategy_id: str, tenant_id: str) -> SelfHealingStrategyPerformance | None:
        return self._profiles.get((strategy_id, tenant_id))

    def put(self, profile: SelfHealingStrategyPerformance) -> SelfHealingStrategyPerformance:
        self._profiles[(profile.strategy_id, profile.tenant_id)] = profile
        return profile


class SelfHealingStrategyPerformanceEngine:
    def __init__(
        self,
        performance_store: InMemorySelfHealingStrategyPerformanceStore,
        quality_store: InMemorySelfHealingStrategyQualityStore | None = None,
    ) -> None:
        self._performance = performance_store
        self._quality = quality_store

    def apply_workflow_outcome(
        self,
        outcome: SelfHealingWorkflowOutcome,
        *,
        validation_status: ValidationDecisionStatus | None = None,
    ) -> SelfHealingStrategyPerformance:
        current = self._performance.get(
            strategy_id=outcome.strategy_id,
            tenant_id=outcome.tenant_id,
        )
        executions = (current.executions if current else 0) + 1
        successes = int((current.success_rate * (executions - 1)) if current else 0)
        rollbacks = int((current.rollback_rate * (executions - 1)) if current else 0)
        total_recovery = (current.average_recovery_time_seconds * (executions - 1)) if current else 0.0

        passed = validation_status is ValidationDecisionStatus.PASSED or (
            validation_status is None
            and outcome.validation_result is not None
            and outcome.validation_result.status.value == "PASSED"
        )
        if passed:
            successes += 1
        if outcome.rollback_executed:
            rollbacks += 1
        total_recovery += outcome.recovery_time.total_seconds()

        profile = SelfHealingStrategyPerformance(
            strategy_id=outcome.strategy_id,
            tenant_id=outcome.tenant_id,
            executions=executions,
            success_rate=successes / executions,
            rollback_rate=rollbacks / executions,
            average_recovery_time_seconds=total_recovery / executions,
            confidence_calibration=0.85 if passed else 0.4,
        )
        return self._performance.put(profile)


__all__ = [
    "InMemorySelfHealingStrategyPerformanceStore",
    "SelfHealingStrategyPerformanceEngine",
]
