# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Immutable versioned predictive analysis context (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from intergrax.contracts.predictive.completeness import PredictiveContextCompleteness
from intergrax.contracts.predictive.provenance import PredictiveContextProvenance
from intergrax.contracts.predictive.scope import PredictiveScope
from intergrax.contracts.predictive.sections import (
    PredictiveContextDiagnostic,
    PredictiveContextHistory,
    PredictiveContextPerformance,
)
from intergrax.contracts.predictive.types import (
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
)
from intergrax.contracts.predictive_historical_intelligence import HistoricalRiskIntelligence

PREDICTIVE_CONTEXT_VERSION = "predictive.context@4.0.0"
_MAX_BOUNDED_TUPLE = 10_000


@dataclass(frozen=True, slots=True)
class PredictiveContextMetadata:
    generated_at: datetime
    context_version: str
    completeness: PredictiveContextCompleteness
    missing_providers: tuple[str, ...] = ()
    input_snapshot_id: str = ""
    context_snapshot_id: str = ""
    provenance: tuple[PredictiveContextProvenance, ...] = ()

    def __post_init__(self) -> None:
        if not self.context_version.strip():
            raise ValueError("context_version must be non-empty")
        if not self.input_snapshot_id.strip():
            raise ValueError("input_snapshot_id must be non-empty")
        if not self.context_snapshot_id.strip():
            object.__setattr__(self, "context_snapshot_id", self.input_snapshot_id)


@dataclass(frozen=True, slots=True)
class PredictiveContext:
    """
    Readonly inputs for predictive analyzers.

    Must never be used to mint or mutate Problems.
    """

    scope: PredictiveScope
    history: PredictiveContextHistory
    performance: PredictiveContextPerformance
    diagnostic: PredictiveContextDiagnostic
    metadata: PredictiveContextMetadata
    current_state: tuple[str, ...] = ()
    decision_history: tuple[str, ...] = ()
    lineage_patterns: tuple[str, ...] = ()
    historical_risk_intelligence: HistoricalRiskIntelligence = field(
        default_factory=HistoricalRiskIntelligence,
    )

    def __post_init__(self) -> None:
        _assert_bounded("history.execution_patterns", self.history.execution_patterns)
        _assert_bounded("performance.latency_series", self.performance.latency_series)
        _assert_bounded("diagnostic.historical_problems", self.diagnostic.historical_problems)

    @property
    def tenant_id(self) -> str:
        return self.scope.tenant_id

    @property
    def execution_patterns(self) -> tuple[ExecutionPatternSnapshot, ...]:
        return self.history.execution_patterns

    @property
    def failure_history(self) -> tuple[PerformanceMetricPoint, ...]:
        return self.history.failure_patterns

    @property
    def performance_history(self) -> tuple[PerformanceMetricPoint, ...]:
        return (
            self.performance.latency_series
            + self.performance.throughput_series
            + self.performance.resource_signals
            + self.history.latency_patterns
            + self.history.retry_patterns
        )

    @property
    def historical_problems(self) -> tuple[HistoricalProblemRef, ...]:
        return self.diagnostic.historical_problems

    @property
    def input_snapshot_id(self) -> str:
        return self.metadata.input_snapshot_id

    @property
    def context_snapshot_id(self) -> str:
        return self.metadata.context_snapshot_id

    @property
    def provenance(self) -> tuple[PredictiveContextProvenance, ...]:
        return self.metadata.provenance

    @property
    def as_of(self) -> datetime:
        return self.metadata.generated_at

    @property
    def completeness(self) -> PredictiveContextCompleteness:
        return self.metadata.completeness

    def with_historical_risk_intelligence(
        self,
        intelligence: HistoricalRiskIntelligence,
    ) -> PredictiveContext:
        return replace(self, historical_risk_intelligence=intelligence)

    def to_serializable_mapping(self) -> dict[str, Any]:
        """JSON-friendly mapping — datetimes as ISO-8601 strings."""

        def _point(p: PerformanceMetricPoint) -> dict[str, Any]:
            return {
                "metric_name": p.metric_name,
                "value": p.value,
                "observed_at": p.observed_at.isoformat(),
                "component_id": p.component_id,
            }

        def _pattern(p: ExecutionPatternSnapshot) -> dict[str, Any]:
            return {
                "subject_identity": p.subject_identity,
                "execution_count": p.execution_count,
                "failed_execution_count": p.failed_execution_count,
                "avg_latency_ms": p.avg_latency_ms,
                "timeout_count": p.timeout_count,
                "token_usage_total": p.token_usage_total,
            }

        return {
            "scope": {
                "tenant_id": self.scope.tenant_id,
                "task_id": self.scope.task_id,
                "run_id": self.scope.run_id,
                "execution_id": self.scope.execution_id,
            },
            "history": {
                "execution_patterns": [_pattern(p) for p in self.history.execution_patterns],
                "failure_patterns": [_point(p) for p in self.history.failure_patterns],
                "retry_patterns": [_point(p) for p in self.history.retry_patterns],
                "latency_patterns": [_point(p) for p in self.history.latency_patterns],
            },
            "performance": {
                "latency_series": [_point(p) for p in self.performance.latency_series],
                "throughput_series": [_point(p) for p in self.performance.throughput_series],
                "resource_signals": [_point(p) for p in self.performance.resource_signals],
            },
            "diagnostic": {
                "previous_findings": [
                    {"finding_id": f.finding_id, "summary": f.summary}
                    for f in self.diagnostic.previous_findings
                ],
                "previous_risk_signals": [
                    {"signal_id": s.signal_id, "summary": s.summary}
                    for s in self.diagnostic.previous_risk_signals
                ],
                "historical_problems": [
                    {
                        "problem_id": h.problem_id,
                        "observed_at": h.observed_at.isoformat(),
                        "summary": h.summary,
                    }
                    for h in self.diagnostic.historical_problems
                ],
            },
            "metadata": {
                "generated_at": self.metadata.generated_at.isoformat(),
                "context_version": self.metadata.context_version,
                "completeness": self.metadata.completeness.value,
                "missing_providers": list(self.metadata.missing_providers),
                "input_snapshot_id": self.metadata.input_snapshot_id,
                "context_snapshot_id": self.metadata.context_snapshot_id,
                "provenance": [
                    {
                        "source": p.source,
                        "version": p.version,
                        "generated_at": p.generated_at.isoformat(),
                        "tenant_scope": p.tenant_scope,
                    }
                    for p in self.metadata.provenance
                ],
            },
            "current_state": list(self.current_state),
            "decision_history": list(self.decision_history),
            "lineage_patterns": list(self.lineage_patterns),
        }


def _assert_bounded(label: str, values: tuple[Any, ...]) -> None:
    if len(values) > _MAX_BOUNDED_TUPLE:
        raise ValueError(f"{label} exceeds bounded limit {_MAX_BOUNDED_TUPLE}")


def predictive_context_from_legacy_fields(
    *,
    tenant_id: str,
    current_state: tuple[str, ...],
    historical_problems: tuple[HistoricalProblemRef, ...],
    execution_patterns: tuple[ExecutionPatternSnapshot, ...],
    failure_history: tuple[PerformanceMetricPoint, ...],
    performance_history: tuple[PerformanceMetricPoint, ...],
    decision_history: tuple[str, ...],
    lineage_patterns: tuple[str, ...],
    input_snapshot_id: str,
    as_of: datetime,
    historical_risk_intelligence: HistoricalRiskIntelligence | None = None,
    completeness: PredictiveContextCompleteness = PredictiveContextCompleteness.COMPLETE,
    missing_providers: tuple[str, ...] = (),
    task_id: str | None = None,
    run_id: str | None = None,
    execution_id: str | None = None,
) -> PredictiveContext:
    """Construct R4 nested context from R1 flat layout (tests and migration)."""

    latency = tuple(
        p for p in performance_history if p.metric_name == "latency_ms"
    )
    throughput = tuple(
        p for p in performance_history if p.metric_name.startswith("throughput")
    )
    resources = tuple(
        p
        for p in performance_history
        if p.metric_name not in {"latency_ms"} and not p.metric_name.startswith("throughput")
    )
    retry_patterns = tuple(
        p for p in performance_history if p.metric_name == "retry_per_execution"
    )
    if not latency and performance_history:
        latency = performance_history

    scope = PredictiveScope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        execution_id=execution_id,
    )
    history = PredictiveContextHistory(
        execution_patterns=execution_patterns,
        failure_patterns=failure_history,
        retry_patterns=retry_patterns,
        latency_patterns=latency,
    )
    performance = PredictiveContextPerformance(
        latency_series=latency,
        throughput_series=throughput,
        resource_signals=resources,
    )
    diagnostic = PredictiveContextDiagnostic(historical_problems=historical_problems)
    metadata = PredictiveContextMetadata(
        generated_at=as_of,
        context_version=PREDICTIVE_CONTEXT_VERSION,
        completeness=completeness,
        missing_providers=missing_providers,
        input_snapshot_id=input_snapshot_id,
    )
    intelligence = historical_risk_intelligence or HistoricalRiskIntelligence()
    return PredictiveContext(
        scope=scope,
        history=history,
        performance=performance,
        diagnostic=diagnostic,
        metadata=metadata,
        current_state=current_state,
        decision_history=decision_history,
        lineage_patterns=lineage_patterns,
        historical_risk_intelligence=intelligence,
    )


__all__ = [
    "PREDICTIVE_CONTEXT_VERSION",
    "PredictiveContext",
    "PredictiveContextMetadata",
    "predictive_context_from_legacy_fields",
]
