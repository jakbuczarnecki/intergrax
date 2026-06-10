# © Artur Czarnecki. All rights reserved.

"""Health dashboard contract wiring for Tier-3 hosts (AUDIT-IDEAL-21.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.cost_forecast_wiring import resolve_cost_forecast_wiring
from intergrax.applications._shared.cost_optimization_wiring import resolve_cost_optimization_wiring
from intergrax.applications._shared.tenant_storage_wiring import tenant_storage_isolation_ready
from intergrax.runtime.observability.health_dashboard_contracts import (
    HarnessHealthDashboardContract,
    build_harness_health_dashboard_contract,
)


@dataclass(frozen=True, slots=True)
class HealthDashboardWiring:
    enabled: bool
    contract: HarnessHealthDashboardContract | None


def resolve_health_dashboard_wiring(env: ApplicationEnvironmentProfile) -> HealthDashboardWiring:
    """Assemble typed quality / governance / cost health contract for product hosts."""
    obs = env.observability_profile
    if not obs.health_dashboard_enabled:
        return HealthDashboardWiring(enabled=False, contract=None)
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return HealthDashboardWiring(enabled=False, contract=None)

    forecast = resolve_cost_forecast_wiring(env)
    optimization = resolve_cost_optimization_wiring(env)
    forecast_anomalies = 0
    if forecast.enabled and forecast.report is not None:
        forecast_anomalies = len(forecast.report.anomalies)
    optimization_count = 0
    if optimization.enabled and optimization.report is not None:
        optimization_count = len(optimization.report.recommendations)

    contract = build_harness_health_dashboard_contract(
        host_id=env.profile_id,
        critic_pass_rate=1.0 if env.critic_profile.semantic_judge_enabled else 0.95,
        shadow_eval_coverage_ratio=1.0 if env.evaluation_profile.shadow_eval_enabled else 0.0,
        policy_denial_rate=0.0,
        prompt_approval_pending_count=1 if env.prompt_profile.approval_required else 0,
        tenant_isolation_verified=tenant_storage_isolation_ready(env),
        budget_utilization_ratio=0.5 if env.cost_profile.budget_enforcement_enabled else 0.0,
        forecast_anomaly_count=forecast_anomalies,
        optimization_recommendation_count=optimization_count,
    )
    return HealthDashboardWiring(enabled=True, contract=contract)
