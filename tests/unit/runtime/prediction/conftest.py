# © Artur Czarnecki. All rights reserved.

"""Shared predictive R1 test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from intergrax.contracts.predictive_context import (
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
    PredictiveContext,
)

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def crm_agent_showcase_context(
    *,
    tenant_id: str = "tenant-demo",
) -> PredictiveContext:
    """Autonomous Customer Operations — CRM agent degradation showcase."""
    base = _AS_OF - timedelta(days=7)
    latency = tuple(
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=100.0 + i * 15.0,
            observed_at=base + timedelta(days=i),
            component_id="crm_agent",
        )
        for i in range(8)
    )
    return PredictiveContext(
        tenant_id=tenant_id,
        current_state=("workflow:customer_support", "supervisor:autonomous_ops"),
        historical_problems=(
            HistoricalProblemRef(
                problem_id="INC-4521",
                observed_at=base + timedelta(days=2),
                summary="CRM API timeout burst during peak load",
            ),
        ),
        execution_patterns=(
            ExecutionPatternSnapshot(
                subject_identity="crm_agent",
                execution_count=53,
                failed_execution_count=16,
                avg_latency_ms=280.0,
                timeout_count=12,
                token_usage_total=125_000,
            ),
        ),
        failure_history=(),
        performance_history=latency,
        decision_history=(),
        lineage_patterns=("attempt:retrieval", "attempt:pricing", "attempt:crm"),
        input_snapshot_id="showcase_crm_agent",
        as_of=_AS_OF,
    )
