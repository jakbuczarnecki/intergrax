from __future__ import annotations

from intergrax.runtime.architecture.cost_forecast import (
    CostAnomalyRecord,
    CostAnomalySeverity,
)
from intergrax.runtime.architecture.cost_optimization import (
    OptimizationGuardrail,
    build_cost_optimization_report,
)


def test_cost_optimization_marks_non_compliant_high_savings() -> None:
    report = build_cost_optimization_report(
        anomalies=[
            CostAnomalyRecord(
                scope_id="tenant-a",
                severity=CostAnomalySeverity.CRITICAL,
                spend_delta_ratio=0.8,
                token_delta_ratio=0.2,
                reasons=["drift"],
            )
        ],
        guardrails=[
            OptimizationGuardrail(
                guardrail_id="cap",
                description="max savings",
                max_recommended_savings_ratio=0.30,
            )
        ],
    )
    assert report.recommendations
    assert report.recommendations[0].policy_compliant is False


def test_cost_optimization_skips_none_severity() -> None:
    report = build_cost_optimization_report(
        anomalies=[
            CostAnomalyRecord(
                scope_id="tenant-a",
                severity=CostAnomalySeverity.NONE,
                spend_delta_ratio=0.0,
                token_delta_ratio=0.0,
            )
        ],
        guardrails=[],
    )
    assert report.recommendations == []
