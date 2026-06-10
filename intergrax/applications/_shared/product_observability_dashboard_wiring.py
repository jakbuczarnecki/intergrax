# © Artur Czarnecki. All rights reserved.

"""Unified product observability dashboard wiring (AUDIT-IDEAL-5.3 / 21.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter

from intergrax.applications._shared.architecture_health_wiring import resolve_architecture_health_wiring
from intergrax.applications._shared.causal_diagnostics_wiring import resolve_causal_diagnostics_wiring
from intergrax.applications._shared.compliance_profile_wiring import resolve_compliance_profile_wiring
from intergrax.applications._shared.health_dashboard_wiring import resolve_health_dashboard_wiring
from intergrax.applications._shared.product_observability_dashboard_routes import (
    create_product_observability_dashboard_router,
)
from intergrax.applications._shared.strategy_review_wiring import resolve_strategy_review_wiring
from intergrax.applications._shared.tenant_storage_wiring import tenant_storage_isolation_ready
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.observability.product_observability_dashboard import (
    ArchitectureHealthPane,
    CausalDiagnosticsPane,
    GovernanceDashboardPane,
    ProductObservabilityDashboard,
    build_product_observability_dashboard,
)


@dataclass(frozen=True, slots=True)
class ProductObservabilityDashboardWiring:
    enabled: bool
    router: APIRouter | None
    dashboard: ProductObservabilityDashboard | None


def _build_dashboard(
    env: ApplicationEnvironmentProfile,
    *,
    repo_root: Path | None = None,
) -> ProductObservabilityDashboard | None:
    health_wiring = resolve_health_dashboard_wiring(env)
    if not health_wiring.enabled or health_wiring.contract is None:
        return None

    architecture_wiring = resolve_architecture_health_wiring(env)
    causal_wiring = resolve_causal_diagnostics_wiring(env)
    compliance_wiring = resolve_compliance_profile_wiring(env)
    strategy_wiring = resolve_strategy_review_wiring(
        env,
        repo_root=repo_root or Path.cwd(),
    )

    modularity = 0.0
    debt_index = 0.0
    capability_count = 0
    if architecture_wiring.enabled and architecture_wiring.pipeline_report is not None:
        snapshots = architecture_wiring.pipeline_report.snapshots
        if snapshots:
            summary = snapshots[-1].report.summary
            modularity = summary.modularity_score
            debt_index = summary.architecture_debt_index
            capability_count = summary.nodes_total

    causal = CausalDiagnosticsPane(
        run_id="bootstrap",
        link_count=0,
        ready=False,
    )
    if causal_wiring.enabled and causal_wiring.chain is not None:
        causal = CausalDiagnosticsPane(
            run_id=causal_wiring.chain.run_id,
            link_count=len(causal_wiring.chain.links),
            ready=True,
        )

    governance = GovernanceDashboardPane(
        compliance_profile_enabled=compliance_wiring.enabled,
        strategy_review_enabled=strategy_wiring.enabled,
        tenant_isolation_verified=tenant_storage_isolation_ready(env),
        policy_denial_rate=health_wiring.contract.governance.policy_denial_rate,
        prompt_approval_pending_count=health_wiring.contract.governance.prompt_approval_pending_count,
    )
    architecture = ArchitectureHealthPane(
        modularity_score=modularity,
        debt_index=debt_index,
        capability_count=capability_count,
    )
    return build_product_observability_dashboard(
        host_id=env.profile_id,
        governance=governance,
        health=health_wiring.contract,
        architecture=architecture,
        causal=causal,
    )


def resolve_product_observability_dashboard_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    repo_root: Path | None = None,
) -> ProductObservabilityDashboardWiring:
    """Mount GOV-PROD.1 dashboard routes when product observability flags are enabled."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ProductObservabilityDashboardWiring(enabled=False, router=None, dashboard=None)

    obs = env.observability_profile
    gov = env.governance_profile
    if not obs.unified_observability_dashboard_enabled and not gov.governance_dashboard_enabled:
        return ProductObservabilityDashboardWiring(enabled=False, router=None, dashboard=None)

    dashboard = _build_dashboard(env, repo_root=repo_root)
    if dashboard is None:
        return ProductObservabilityDashboardWiring(enabled=False, router=None, dashboard=None)

    return ProductObservabilityDashboardWiring(
        enabled=True,
        router=create_product_observability_dashboard_router(dashboard=dashboard, enabled=True),
        dashboard=dashboard,
    )
