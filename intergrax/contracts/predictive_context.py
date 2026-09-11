# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Readonly predictive analysis context contract (PREDICTIVE R1)."""

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
class PredictiveContext:
    """
    Readonly inputs for predictive analyzers.

    Must never be used to mint or mutate Problems.
    """

    tenant_id: str
    current_state: tuple[str, ...]
    historical_problems: tuple[HistoricalProblemRef, ...]
    execution_patterns: tuple[ExecutionPatternSnapshot, ...]
    failure_history: tuple[PerformanceMetricPoint, ...]
    performance_history: tuple[PerformanceMetricPoint, ...]
    decision_history: tuple[str, ...]
    lineage_patterns: tuple[str, ...]
    input_snapshot_id: str
    as_of: datetime

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.input_snapshot_id.strip():
            raise ValueError("input_snapshot_id must be non-empty")


__all__ = [
    "ExecutionPatternSnapshot",
    "HistoricalProblemRef",
    "PerformanceMetricPoint",
    "PredictiveContext",
]
