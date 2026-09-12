# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL plugin gateway — strategy invocation only through registry ports."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.compensation_decision import CompensationDecision
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionRequest,
    CompensationPluginExecutionResult,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    CompensationStrategyEvaluationRequest,
    EnterpriseReliabilityPluginRegistry,
    EnterpriseReliabilityStrategyContext,
    ReconciliationStrategyAdvice,
    ResolutionStrategyEvaluationRequest,
    RiskEvaluationStrategyAdvice,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
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

    def resolution_strategy_registered(self, plugin_id: str) -> bool:
        return self._registry.resolve_resolution(plugin_id) is not None

    def evaluate_resolution(
        self,
        plugin_id: str,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None:
        strategy = self._registry.resolve_resolution(plugin_id)
        if strategy is None:
            return None
        return strategy.evaluate(request)

    def compensation_strategy_registered(self, plugin_id: str) -> bool:
        return self._registry.resolve_compensation(plugin_id) is not None

    def evaluate_compensation(
        self,
        plugin_id: str,
        request: CompensationStrategyEvaluationRequest,
    ) -> CompensationDecision | None:
        strategy = self._registry.resolve_compensation(plugin_id)
        if strategy is None:
            return None
        return strategy.evaluate(request)

    def compensation_executor_registered(self, plugin_id: str) -> bool:
        return self._registry.resolve_compensation_executor(plugin_id) is not None

    def execute_compensation(
        self,
        plugin_id: str,
        request: CompensationExecutionRequest,
    ) -> CompensationPluginExecutionResult | None:
        executor = self._registry.resolve_compensation_executor(plugin_id)
        if executor is None:
            return None
        return executor.execute_compensation(request)

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
