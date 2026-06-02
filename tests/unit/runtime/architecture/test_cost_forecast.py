from __future__ import annotations

from intergrax.runtime.architecture.cost_forecast import (
    CostAnomalySeverity,
    CostUsageSnapshot,
    build_cost_forecast_report,
)


def test_cost_forecast_detects_critical_anomaly() -> None:
    report = build_cost_forecast_report(
        baseline=[
            CostUsageSnapshot(scope_id="tenant-a", spend_amount=100.0, token_count=10_000)
        ],
        current=[
            CostUsageSnapshot(scope_id="tenant-a", spend_amount=200.0, token_count=20_000)
        ],
        critical_ratio=0.50,
    )
    assert report.anomalies
    assert report.anomalies[0].severity == CostAnomalySeverity.CRITICAL


def test_cost_forecast_builds_projection_points() -> None:
    report = build_cost_forecast_report(
        baseline=[CostUsageSnapshot(scope_id="tenant-a", spend_amount=100.0, token_count=1000)],
        current=[CostUsageSnapshot(scope_id="tenant-a", spend_amount=110.0, token_count=1100)],
    )
    assert report.forecasts[0].projected_spend > 110.0
