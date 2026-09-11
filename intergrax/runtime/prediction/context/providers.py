# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Static in-memory predictive context providers (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive import (
    PredictiveContextDiagnostic,
    PredictiveContextHistory,
    PredictiveContextPerformance,
    PredictiveContextProviderFragment,
    PredictiveContextProviderStatus,
    PredictiveScope,
)


@dataclass(frozen=True, slots=True)
class ExecutionHistoryProvider:
    provider_id: str = "execution_history"
    provider_version: str = "v1"
    history: PredictiveContextHistory | None = None

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        if scope.tenant_id != scope.tenant_id.strip():
            raise ValueError("tenant_id required")
        return PredictiveContextProviderFragment(
            provider_id=self.provider_id,
            status=PredictiveContextProviderStatus.SUCCESS,
            history=self.history or PredictiveContextHistory(),
        )


@dataclass(frozen=True, slots=True)
class FailureHistoryProvider:
    provider_id: str = "failure_history"
    provider_version: str = "v1"
    failure_patterns: PredictiveContextHistory | None = None

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        history = self.failure_patterns or PredictiveContextHistory()
        return PredictiveContextProviderFragment(
            provider_id=self.provider_id,
            status=PredictiveContextProviderStatus.SUCCESS,
            history=PredictiveContextHistory(failure_patterns=history.failure_patterns),
        )


@dataclass(frozen=True, slots=True)
class PerformanceHistoryProvider:
    provider_id: str = "performance_history"
    provider_version: str = "v2"
    performance: PredictiveContextPerformance | None = None

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        perf = self.performance or PredictiveContextPerformance()
        return PredictiveContextProviderFragment(
            provider_id=self.provider_id,
            status=PredictiveContextProviderStatus.SUCCESS,
            performance=perf,
        )


@dataclass(frozen=True, slots=True)
class DecisionHistoryProvider:
    provider_id: str = "decision_history"
    provider_version: str = "v1"
    decision_history: tuple[str, ...] = ()

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        return PredictiveContextProviderFragment(
            provider_id=self.provider_id,
            status=PredictiveContextProviderStatus.SUCCESS,
            decision_history=self.decision_history,
        )


@dataclass(frozen=True, slots=True)
class BusinessSignalProvider:
    provider_id: str = "business_metrics"
    provider_version: str = "v1"
    current_state: tuple[str, ...] = ()

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        return PredictiveContextProviderFragment(
            provider_id=self.provider_id,
            status=PredictiveContextProviderStatus.SUCCESS,
            current_state=self.current_state,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticHistoryProvider:
    provider_id: str = "diagnostic_history"
    provider_version: str = "v1"
    diagnostic: PredictiveContextDiagnostic | None = None
    lineage_patterns: tuple[str, ...] = ()
    current_state: tuple[str, ...] = ()

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        return PredictiveContextProviderFragment(
            provider_id=self.provider_id,
            status=PredictiveContextProviderStatus.SUCCESS,
            diagnostic=self.diagnostic or PredictiveContextDiagnostic(),
            lineage_patterns=self.lineage_patterns,
            current_state=self.current_state,
        )


__all__ = [
    "BusinessSignalProvider",
    "DecisionHistoryProvider",
    "DiagnosticHistoryProvider",
    "ExecutionHistoryProvider",
    "FailureHistoryProvider",
    "PerformanceHistoryProvider",
]
