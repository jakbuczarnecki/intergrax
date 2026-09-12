# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Observe completed workflows — append-only memory, no execution authority (SELF-HEALING R5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext
from intergrax.contracts.self_healing.performance_memory.record import (
    SelfHealingStrategyPerformanceExperience,
    mint_self_healing_strategy_performance_experience_id,
)
from intergrax.contracts.self_healing.performance_memory.repository import StrategyPerformanceMemoryRepository
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.runtime.self_healing.performance_memory.outcome_derivation import derive_strategy_execution_outcome


@dataclass
class StrategyPerformanceMemoryRecorder:
    repository: StrategyPerformanceMemoryRepository

    def observe_workflow_completion(
        self,
        outcome: SelfHealingWorkflowOutcome,
        execution_context: SelfHealingExecutionContext,
        *,
        diagnostic_investigation_id: str,
        recorded_at: datetime | None = None,
    ) -> SelfHealingStrategyPerformanceExperience:
        if outcome.workflow_id != execution_context.workflow_id:
            raise ValueError("workflow_id mismatch between outcome and execution context")
        if outcome.strategy_id != execution_context.strategy_id:
            raise ValueError("strategy_id mismatch between outcome and execution context")
        if outcome.tenant_id != execution_context.tenant_id:
            raise ValueError("tenant_id mismatch between outcome and execution context")
        if not diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")

        when = recorded_at or datetime.now(timezone.utc)
        experience = SelfHealingStrategyPerformanceExperience(
            experience_id=mint_self_healing_strategy_performance_experience_id(),
            tenant_id=outcome.tenant_id,
            strategy_id=outcome.strategy_id,
            workflow_id=outcome.workflow_id,
            plan_id=execution_context.plan_id,
            execution_ids=execution_context.execution_ids,
            diagnostic_investigation_id=diagnostic_investigation_id,
            execution_outcome=derive_strategy_execution_outcome(outcome),
            rollback_executed=outcome.rollback_executed,
            recovery_time_seconds=outcome.recovery_time.total_seconds(),
            evidence_refs=outcome.evidence_refs,
            recorded_at=when,
        )
        return self.repository.append(experience)


__all__ = ["StrategyPerformanceMemoryRecorder"]
