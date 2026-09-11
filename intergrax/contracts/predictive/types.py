# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Shared predictive context value types (PREDICTIVE R1/R4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PerformanceMetricPoint:
    """One observed performance sample — not diagnostic truth."""

    metric_name: str
    value: float
    observed_at: datetime
    component_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPatternSnapshot:
    """Aggregated execution behavior for one subject — evidence-derived only."""

    subject_identity: str
    execution_count: int
    failed_execution_count: int
    avg_latency_ms: float | None
    timeout_count: int = 0
    token_usage_total: int | None = None


@dataclass(frozen=True, slots=True)
class HistoricalProblemRef:
    """Pointer to a prior Problem — correlation only, not causation."""

    problem_id: str
    observed_at: datetime
    summary: str


@dataclass(frozen=True, slots=True)
class PredictiveFindingRef:
    """Read-only prior diagnostic finding reference."""

    finding_id: str
    summary: str


@dataclass(frozen=True, slots=True)
class PredictiveRiskSignalRef:
    """Read-only prior risk signal reference — not a Problem."""

    signal_id: str
    summary: str


__all__ = [
    "ExecutionPatternSnapshot",
    "HistoricalProblemRef",
    "PerformanceMetricPoint",
    "PredictiveFindingRef",
    "PredictiveRiskSignalRef",
]
