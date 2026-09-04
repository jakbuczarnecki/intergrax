# © Artur Czarnecki. All rights reserved.

"""Health dashboard contract wiring for Tier-3 hosts (AUDIT-IDEAL-21.2 / DIAG-FOUNDATION-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications._shared.auditability_health_wiring import (
    HostAuditabilityHealthFacts,
    project_auditability_health_snapshot,
    project_conservative_auditability_health_facts,
    project_host_auditability_health_facts_from_runtime,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.cost_forecast_wiring import resolve_cost_forecast_wiring
from intergrax.applications._shared.cost_optimization_wiring import resolve_cost_optimization_wiring
from intergrax.applications._shared.tenant_storage_wiring import tenant_storage_isolation_ready
from intergrax.runtime.observability.health_dashboard_contracts import (
    HarnessHealthDashboardContract,
    build_harness_health_dashboard_contract,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


@dataclass(frozen=True, slots=True)
class HealthDashboardWiring:
    enabled: bool
    contract: HarnessHealthDashboardContract | None


def _resolve_auditability_facts(
    env: ApplicationEnvironmentProfile,
    *,
    auditability_facts: HostAuditabilityHealthFacts | None = None,
) -> HostAuditabilityHealthFacts:
    if auditability_facts is not None:
        return auditability_facts
    return project_conservative_auditability_health_facts(env)


def resolve_health_dashboard_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    auditability_facts: HostAuditabilityHealthFacts | None = None,
) -> HealthDashboardWiring:
    """Assemble typed quality / governance / cost / auditability health contract."""
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

    facts = _resolve_auditability_facts(env, auditability_facts=auditability_facts)
    contract = build_harness_health_dashboard_contract(
        host_id=env.profile_id,
        auditability=project_auditability_health_snapshot(facts),
        shadow_eval_coverage_ratio=1.0 if env.evaluation_profile.shadow_eval_enabled else 0.0,
        policy_denial_rate=0.0,
        prompt_approval_pending_count=1 if env.prompt_profile.approval_required else 0,
        tenant_isolation_verified=tenant_storage_isolation_ready(env),
        budget_utilization_ratio=0.5 if env.cost_profile.budget_enforcement_enabled else 0.0,
        forecast_anomaly_count=forecast_anomalies,
        optimization_recommendation_count=optimization_count,
    )
    return HealthDashboardWiring(enabled=True, contract=contract)


def resolve_health_dashboard_wiring_from_runtime(
    runtime: HarnessHostRuntime,
    *,
    diagnostic_read_side_ready: bool,
) -> HealthDashboardWiring:
    """Runtime-aware health assembly using ``HarnessHostRuntime.diagnostic_wiring``."""
    facts = project_host_auditability_health_facts_from_runtime(
        runtime,
        diagnostic_read_side_ready=diagnostic_read_side_ready,
    )
    return resolve_health_dashboard_wiring(
        runtime.environment,
        auditability_facts=facts,
    )
