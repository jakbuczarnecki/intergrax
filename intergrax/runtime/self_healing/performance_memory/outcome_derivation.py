# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Map workflow outcomes to factual execution outcome labels (SELF-HEALING R5.1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyExecutionOutcome
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.contracts.self_healing.workflow.validation import ValidationStatus


def derive_strategy_execution_outcome(
    outcome: SelfHealingWorkflowOutcome,
) -> SelfHealingStrategyExecutionOutcome:
    if outcome.rollback_executed:
        return SelfHealingStrategyExecutionOutcome.ROLLED_BACK
    validation = outcome.validation_result
    if validation is None:
        return SelfHealingStrategyExecutionOutcome.INCONCLUSIVE
    if validation.status is ValidationStatus.PASSED:
        return SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED
    if validation.status is ValidationStatus.FAILED:
        return SelfHealingStrategyExecutionOutcome.REPAIR_FAILED
    return SelfHealingStrategyExecutionOutcome.INCONCLUSIVE


__all__ = ["derive_strategy_execution_outcome"]
