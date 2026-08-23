# © Artur Czarnecki. All rights reserved.

"""Explicit control-plane governance wiring for reference production lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


@dataclass(frozen=True, slots=True)
class ReferenceProductionControlPlaneGovernance:
    """Trusted principal + explicit tenant authority + configured mutation policy."""

    principal: RequestIdentity
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary
    environment_tenant_resolver: ApplicationEnvironmentTenantResolver


class _ReferenceProductionConfiguredAllowEvaluator:
    """Explicit reference-host policy: configured service principal may activate."""

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        del request
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="reference_production_policy_configured",
            policy_rule_id="reference_production.lifecycle_activate",
        )


def build_reference_production_control_plane_governance(
    env: ApplicationEnvironmentProfile,
    *,
    service_id: str = "reference-production-lifecycle",
    tenant_id: str | None = None,
) -> ReferenceProductionControlPlaneGovernance:
    """Build explicit single-tenant reference production governance from host profile."""
    resolved_tenant = (tenant_id or env.profile_id).strip()
    if not resolved_tenant:
        raise ValueError("reference production tenant authority requires profile_id")
    principal = RequestIdentity(
        tenant_id=resolved_tenant,
        user_id=service_id,
        principal_type=PrincipalType.SERVICE,
        auth_subject=service_id,
    )
    return ReferenceProductionControlPlaneGovernance(
        principal=principal,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_ReferenceProductionConfiguredAllowEvaluator(),
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=resolved_tenant,
        ),
    )


def wire_governed_reference_production_launcher(
    composition: ProductionProcessComposition,
    env: ApplicationEnvironmentProfile,
    *,
    service_id: str = "reference-production-lifecycle",
    tenant_id: str | None = None,
) -> tuple[ReferenceProductionLifecycleLauncher, ReferenceProductionControlPlaneGovernance]:
    """Return launcher + governance bundle for one explicit reference production host."""
    governance = build_reference_production_control_plane_governance(
        env,
        service_id=service_id,
        tenant_id=tenant_id,
    )
    launcher = ReferenceProductionLifecycleLauncher(
        composition,
        mutation_authorization_boundary=governance.mutation_authorization_boundary,
        environment_tenant_resolver=governance.environment_tenant_resolver,
    )
    return launcher, governance


__all__ = [
    "ReferenceProductionControlPlaneGovernance",
    "build_reference_production_control_plane_governance",
    "wire_governed_reference_production_launcher",
]
