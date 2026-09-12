# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Workflow outcome → strategy quality (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.quality import SelfHealingStrategyQualityProfile
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.contracts.self_healing.workflow.validation import ValidationStatus
from intergrax.runtime.self_healing.outcome_learning import InMemorySelfHealingStrategyQualityStore


class SelfHealingWorkflowOutcomeEngine:
    def __init__(self, store: InMemorySelfHealingStrategyQualityStore) -> None:
        self._store = store

    def apply_workflow_outcome(self, outcome: SelfHealingWorkflowOutcome) -> SelfHealingStrategyQualityProfile:
        current = self._store.get(strategy_id=outcome.strategy_id, tenant_id=outcome.tenant_id)
        if current is None:
            current = SelfHealingStrategyQualityProfile(
                strategy_id=outcome.strategy_id,
                tenant_id=outcome.tenant_id,
            )
        successful = current.successful_preventions
        failed = current.failed_actions
        rollback_rate = current.rollback_rate
        if outcome.validation_result and outcome.validation_result.status is ValidationStatus.PASSED:
            successful += 1
        else:
            failed += 1
        if outcome.rollback_executed:
            rollback_rate = min(1.0, rollback_rate + 0.1)
        updated = SelfHealingStrategyQualityProfile(
            strategy_id=outcome.strategy_id,
            tenant_id=outcome.tenant_id,
            successful_preventions=successful,
            failed_actions=failed,
            rollback_rate=rollback_rate,
            false_positive_rate=current.false_positive_rate,
        )
        return self._store.put(updated)


__all__ = ["SelfHealingWorkflowOutcomeEngine"]
