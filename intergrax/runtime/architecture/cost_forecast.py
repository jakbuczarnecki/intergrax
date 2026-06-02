# © Artur Czarnecki. All rights reserved.

"""Spend and token drift forecast with anomaly detection (Phase V-COST.3)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CostAnomalySeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


class CostUsageSnapshot(BaseModel):
    scope_id: str
    spend_amount: float
    token_count: int


class CostForecastPoint(BaseModel):
    scope_id: str
    projected_spend: float
    projected_tokens: int


class CostAnomalyRecord(BaseModel):
    scope_id: str
    severity: CostAnomalySeverity
    spend_delta_ratio: float
    token_delta_ratio: float
    reasons: list[str] = Field(default_factory=list)


class CostForecastReport(BaseModel):
    schema_version: str = "1.0.0"
    baseline: list[CostUsageSnapshot] = Field(default_factory=list)
    current: list[CostUsageSnapshot] = Field(default_factory=list)
    forecasts: list[CostForecastPoint] = Field(default_factory=list)
    anomalies: list[CostAnomalyRecord] = Field(default_factory=list)


def build_cost_forecast_report(
    *,
    baseline: list[CostUsageSnapshot],
    current: list[CostUsageSnapshot],
    growth_multiplier: float = 1.15,
    warning_ratio: float = 0.25,
    critical_ratio: float = 0.50,
) -> CostForecastReport:
    baseline_by_scope = {snapshot.scope_id: snapshot for snapshot in baseline}
    current_by_scope = {snapshot.scope_id: snapshot for snapshot in current}
    forecasts: list[CostForecastPoint] = []
    anomalies: list[CostAnomalyRecord] = []

    for scope_id, current_snapshot in current_by_scope.items():
        forecasts.append(
            CostForecastPoint(
                scope_id=scope_id,
                projected_spend=current_snapshot.spend_amount * growth_multiplier,
                projected_tokens=int(float(current_snapshot.token_count) * growth_multiplier),
            )
        )
        baseline_snapshot = baseline_by_scope.get(scope_id)
        if baseline_snapshot is None:
            continue
        spend_delta_ratio = _delta_ratio(
            baseline_snapshot.spend_amount,
            current_snapshot.spend_amount,
        )
        token_delta_ratio = _delta_ratio(
            float(baseline_snapshot.token_count),
            float(current_snapshot.token_count),
        )
        severity = CostAnomalySeverity.NONE
        reasons: list[str] = []
        if spend_delta_ratio >= critical_ratio or token_delta_ratio >= critical_ratio:
            severity = CostAnomalySeverity.CRITICAL
            reasons.append("Critical spend or token drift detected")
        elif spend_delta_ratio >= warning_ratio or token_delta_ratio >= warning_ratio:
            severity = CostAnomalySeverity.WARNING
            reasons.append("Warning-level spend or token drift detected")
        if severity != CostAnomalySeverity.NONE:
            anomalies.append(
                CostAnomalyRecord(
                    scope_id=scope_id,
                    severity=severity,
                    spend_delta_ratio=spend_delta_ratio,
                    token_delta_ratio=token_delta_ratio,
                    reasons=reasons,
                )
            )

    return CostForecastReport(
        baseline=baseline,
        current=current,
        forecasts=forecasts,
        anomalies=anomalies,
    )


def _delta_ratio(previous: float, current: float) -> float:
    if previous <= 0.0:
        return 0.0 if current <= 0.0 else 1.0
    return abs(current - previous) / previous
