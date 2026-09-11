# © Artur Czarnecki. All rights reserved.

"""Shared predictive R1 test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from intergrax.contracts.predictive_context import (
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
    PredictiveContext,
    predictive_context_from_legacy_fields,
)

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def crm_agent_showcase_context(
    *,
    tenant_id: str = "tenant-demo",
) -> "PredictiveContext":
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
    return predictive_context_from_legacy_fields(
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


def crm_agent_incident_prevention_context(
    *,
    tenant_id: str = "tenant-demo",
) -> "PredictiveContext":
    """CRM Agent Incident Prevention — R4 enterprise reference scenario."""
    base = _AS_OF - timedelta(days=14)
    latency = (
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=100.0,
            observed_at=base,
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=120.0,
            observed_at=base + timedelta(days=5),
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=135.0,
            observed_at=base + timedelta(days=10),
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=180.0,
            observed_at=base + timedelta(days=13),
            component_id="crm_agent",
        ),
    )
    retry = (
        PerformanceMetricPoint(
            metric_name="retry_per_execution",
            value=0.02,
            observed_at=base,
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="retry_per_execution",
            value=0.05,
            observed_at=base + timedelta(days=7),
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="retry_per_execution",
            value=0.18,
            observed_at=base + timedelta(days=13),
            component_id="crm_agent",
        ),
    )
    failure = (
        PerformanceMetricPoint(
            metric_name="failure_count",
            value=0.001,
            observed_at=base,
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="failure_count",
            value=0.01,
            observed_at=base + timedelta(days=7),
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="failure_count",
            value=0.07,
            observed_at=base + timedelta(days=13),
            component_id="crm_agent",
        ),
    )
    performance = latency + retry + failure
    return predictive_context_from_legacy_fields(
        tenant_id=tenant_id,
        current_state=("workflow:crm_agent", "supervisor:autonomous_ops"),
        historical_problems=(
            HistoricalProblemRef(
                problem_id="INC-CRM-991",
                observed_at=base + timedelta(days=6),
                summary="CRM API latency spike",
            ),
        ),
        execution_patterns=(
            ExecutionPatternSnapshot(
                subject_identity="crm_agent",
                execution_count=10_000,
                failed_execution_count=700,
                avg_latency_ms=180.0,
                timeout_count=1500,
            ),
        ),
        failure_history=failure,
        performance_history=performance,
        decision_history=(),
        lineage_patterns=("attempt:crm_lookup", "attempt:pricing"),
        input_snapshot_id="showcase_crm_incident_prevention",
        as_of=_AS_OF,
    )


def crm_agent_day2_future_evidence() -> "PredictionFutureEvidenceSnapshot":
    """Day 2 — CRM API unavailable; diagnostic history records Problem (read-only facts)."""
    from datetime import timedelta

    from intergrax.runtime.prediction.outcome.prediction_outcome_resolver import (
        PredictionFutureEvidenceSnapshot,
    )

    return PredictionFutureEvidenceSnapshot(
        observed_at=_AS_OF + timedelta(days=1),
        evidence_refs=(
            "execution_pattern:crm_agent",
            "event:EXECUTION_FAILED:crm_timeout_spike",
        ),
        execution_failed=True,
        problem_created_for_subject=True,
        matching_risk_keywords=("crm_agent", "crm", "degradation_risk"),
    )


def crm_showcase_3_0_governance_timeline() -> dict[str, object]:
    """
    Enterprise CRM Showcase 3.0 — T-60 / T-30 / T-0 / T+10 governance narrative.

    Used by qualification docs; values are illustrative bounded projections.
    """
    return {
        "t_minus_60": {"risk": "MEDIUM", "confidence": 0.72},
        "t_minus_30": {
            "evidence": {"latency_delta": 0.6, "retry_delta": 3.0},
            "risk": "HIGH",
            "confidence": 0.91,
        },
        "t_zero": {"incident": "EXECUTION_FAILED"},
        "t_plus_10": {
            "feedback": "TRUE_POSITIVE",
            "analyzer_quality": "updated",
        },
    }
