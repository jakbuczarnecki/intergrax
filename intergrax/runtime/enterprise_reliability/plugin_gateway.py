# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL plugin gateway — strategy invocation only through registry ports."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.plugin_spi import (
    CompensationStrategyAdvice,
    EnterpriseReliabilityPluginRegistry,
    EnterpriseReliabilityStrategyContext,
    ReconciliationStrategyAdvice,
    ResolutionStrategyAdvice,
    RiskEvaluationStrategyAdvice,
)
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)


class EnterpriseReliabilityPluginGatewayImpl:
    """Default gateway — depends on ``EnterpriseReliabilityPluginRegistry`` port only."""

    def __init__(self, registry: EnterpriseReliabilityPluginRegistry) -> None:
        self._registry = registry

    def evaluate_reconciliation(
        self,
        plugin_id: str,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None:
        strategy = self._registry.resolve_reconciliation(plugin_id)
        if strategy is None:
            return None
        return strategy.evaluate(context)

    def execute_reconciliation_probe(
        self,
        plugin_id: str,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult | None:
        executor = self._registry.resolve_reconciliation_probe(plugin_id)
        if executor is None:
            return None
        return executor.execute_probe(request)

    def evaluate_resolution(
        self,
        plugin_id: str,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ResolutionStrategyAdvice | None:
        strategy = self._registry.resolve_resolution(plugin_id)
        if strategy is None:
            return None
        return strategy.evaluate(context)

    def evaluate_compensation(
        self,
        plugin_id: str,
        context: EnterpriseReliabilityStrategyContext,
    ) -> CompensationStrategyAdvice | None:
        strategy = self._registry.resolve_compensation(plugin_id)
        if strategy is None:
            return None
        return strategy.evaluate(context)

    def evaluate_risk(
        self,
        plugin_id: str,
        context: EnterpriseReliabilityStrategyContext,
    ) -> RiskEvaluationStrategyAdvice | None:
        strategy = self._registry.resolve_risk_evaluation(plugin_id)
        if strategy is None:
            return None
        return strategy.evaluate(context)


__all__ = ["EnterpriseReliabilityPluginGatewayImpl"]
