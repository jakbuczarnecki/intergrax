# © Artur Czarnecki. All rights reserved.

"""Cost optimization recommendations with policy guardrails (Phase V-COST.4)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.cost_forecast import CostAnomalyRecord, CostAnomalySeverity


class OptimizationRecommendationType(str, Enum):
    MODEL_DOWNGRADE = "model_downgrade"
    CACHE_CONTEXT = "cache_context"
    REDUCE_TOOL_FANOUT = "reduce_tool_fanout"
    TIGHTEN_QUOTA = "tighten_quota"


class OptimizationGuardrail(BaseModel):
    guardrail_id: str
    description: str
    max_recommended_savings_ratio: float = 0.30


class OptimizationRecommendation(BaseModel):
    recommendation_type: OptimizationRecommendationType
    scope_id: str
    estimated_savings_ratio: float
    policy_compliant: bool
    reasons: list[str] = Field(default_factory=list)


class CostOptimizationReport(BaseModel):
    schema_version: str = "1.0.0"
    guardrails: list[OptimizationGuardrail] = Field(default_factory=list)
    recommendations: list[OptimizationRecommendation] = Field(default_factory=list)


def build_cost_optimization_report(
    *,
    anomalies: list[CostAnomalyRecord],
    guardrails: list[OptimizationGuardrail],
) -> CostOptimizationReport:
    recommendations: list[OptimizationRecommendation] = []
    max_savings = min(
        (guardrail.max_recommended_savings_ratio for guardrail in guardrails),
        default=0.30,
    )
    for anomaly in anomalies:
        if anomaly.severity == CostAnomalySeverity.NONE:
            continue
        estimated_savings = 0.20 if anomaly.severity == CostAnomalySeverity.WARNING else 0.35
        policy_compliant = estimated_savings <= max_savings
        reasons: list[str] = []
        if not policy_compliant:
            reasons.append("Recommendation exceeds configured savings guardrail")
        recommendations.append(
            OptimizationRecommendation(
                recommendation_type=_recommendation_for_anomaly(anomaly),
                scope_id=anomaly.scope_id,
                estimated_savings_ratio=estimated_savings,
                policy_compliant=policy_compliant,
                reasons=reasons,
            )
        )
    return CostOptimizationReport(guardrails=guardrails, recommendations=recommendations)


def _recommendation_for_anomaly(
    anomaly: CostAnomalyRecord,
) -> OptimizationRecommendationType:
    if anomaly.token_delta_ratio >= anomaly.spend_delta_ratio:
        return OptimizationRecommendationType.CACHE_CONTEXT
    return OptimizationRecommendationType.MODEL_DOWNGRADE
