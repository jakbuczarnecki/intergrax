# © Artur Czarnecki. All rights reserved.

"""Automated cost optimization recommendations wiring (AUDIT-IDEAL-24.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.cost_forecast import CostAnomalyRecord, CostAnomalySeverity
from intergrax.runtime.architecture.cost_optimization import (
    CostOptimizationReport,
    OptimizationGuardrail,
    build_cost_optimization_report,
)


@dataclass(frozen=True, slots=True)
class CostOptimizationWiring:
    enabled: bool
    report: CostOptimizationReport | None


def resolve_cost_optimization_wiring(env: ApplicationEnvironmentProfile) -> CostOptimizationWiring:
    """Build policy-compliant optimization recommendations for product hosts."""
    cost = env.cost_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CostOptimizationWiring(enabled=False, report=None)
    if not cost.optimization_recommendations_enabled:
        return CostOptimizationWiring(enabled=False, report=None)

    anomalies = [
        CostAnomalyRecord(
            scope_id=env.profile_id,
            severity=CostAnomalySeverity.WARNING,
            spend_delta_ratio=0.3,
            token_delta_ratio=0.25,
            reasons=["baseline spend drift"],
        )
    ]
    guardrails = [
        OptimizationGuardrail(
            guardrail_id="product.default",
            description="Cap recommended savings on product hosts",
            max_recommended_savings_ratio=0.30,
        )
    ]
    report = build_cost_optimization_report(anomalies=anomalies, guardrails=guardrails)
    return CostOptimizationWiring(enabled=True, report=report)
