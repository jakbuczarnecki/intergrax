# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Contained strategy evaluation — plugin isolation (SELF-HEALING R1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.result import (
    SelfHealingStrategyEvaluationResult,
    SelfHealingStrategyEvaluationStatus,
)
from intergrax.contracts.self_healing.safety import assert_strategy_has_no_execution_surface
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy
from intergrax.runtime.self_healing.resolution import resolve_strategies_for_context
from intergrax.runtime.self_healing.strategy_registry import InMemorySelfHealingStrategyRegistry


class SelfHealingDecisionEngine:
    """Central decision layer — evaluates registered strategies with containment."""

    def __init__(self, registry: InMemorySelfHealingStrategyRegistry) -> None:
        self._registry = registry

    def evaluate_contained(
        self,
        strategy: SelfHealingStrategy,
        context: SelfHealingContext,
    ) -> SelfHealingStrategyEvaluationResult:
        assert_strategy_has_no_execution_surface(strategy)
        try:
            decision = strategy.evaluate(context)
        except Exception as exc:  # noqa: BLE001 — plugin containment boundary
            return SelfHealingStrategyEvaluationResult(
                strategy_id=strategy.strategy_id,
                status=SelfHealingStrategyEvaluationStatus.STRATEGY_FAILED,
                failure_reason=str(exc),
            )
        if decision is None:
            return SelfHealingStrategyEvaluationResult(
                strategy_id=strategy.strategy_id,
                status=SelfHealingStrategyEvaluationStatus.ABSTAIN,
            )
        if decision.strategy_id != strategy.strategy_id:
            return SelfHealingStrategyEvaluationResult(
                strategy_id=strategy.strategy_id,
                status=SelfHealingStrategyEvaluationStatus.STRATEGY_FAILED,
                failure_reason="decision.strategy_id mismatch",
            )
        return SelfHealingStrategyEvaluationResult(
            strategy_id=strategy.strategy_id,
            status=SelfHealingStrategyEvaluationStatus.DECISION,
            decision=decision,
        )

    def select_and_evaluate(
        self,
        context: SelfHealingContext,
    ) -> tuple[SelfHealingStrategyEvaluationResult, ...]:
        strategies = self._registry.list_available(tenant_id=context.tenant_id)
        ordered = resolve_strategies_for_context(strategies, context)
        results: list[SelfHealingStrategyEvaluationResult] = []
        for strategy in ordered:
            result = self.evaluate_contained(strategy, context)
            results.append(result)
            if result.status is SelfHealingStrategyEvaluationStatus.DECISION:
                break
        return tuple(results)


__all__ = ["SelfHealingDecisionEngine"]
