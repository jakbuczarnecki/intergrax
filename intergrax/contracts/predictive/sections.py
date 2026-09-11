# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Nested predictive context sections (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive.types import (
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
    PredictiveFindingRef,
    PredictiveRiskSignalRef,
)


@dataclass(frozen=True, slots=True)
class PredictiveContextHistory:
    execution_patterns: tuple[ExecutionPatternSnapshot, ...] = ()
    failure_patterns: tuple[PerformanceMetricPoint, ...] = ()
    retry_patterns: tuple[PerformanceMetricPoint, ...] = ()
    latency_patterns: tuple[PerformanceMetricPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class PredictiveContextPerformance:
    latency_series: tuple[PerformanceMetricPoint, ...] = ()
    throughput_series: tuple[PerformanceMetricPoint, ...] = ()
    resource_signals: tuple[PerformanceMetricPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class PredictiveContextDiagnostic:
    previous_findings: tuple[PredictiveFindingRef, ...] = ()
    previous_risk_signals: tuple[PredictiveRiskSignalRef, ...] = ()
    historical_problems: tuple[HistoricalProblemRef, ...] = ()


__all__ = [
    "PredictiveContextDiagnostic",
    "PredictiveContextHistory",
    "PredictiveContextPerformance",
]
