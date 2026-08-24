# © Artur Czarnecki. All rights reserved.

"""Explicit control-plane governance wiring for production capacity probe."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.runtime.capacity.control_plane_governance import (
    EcpResourceTenantResolver,
    StaticEcpResourceTenantResolver,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


@dataclass(frozen=True, slots=True)
class ProductionCapacityGovernance:
    """Trusted principal + explicit tenant authority + configured mutation policy."""

    principal: RequestIdentity
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None
    tenant_resolver: EcpResourceTenantResolver
    tenant_id: str


def build_production_capacity_governance(
    env: ApplicationEnvironmentProfile,
    *,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    service_id: str = "production-capacity-probe",
    tenant_id: str | None = None,
) -> ProductionCapacityGovernance:
    """Build production capacity governance shell; policy must be supplied by composition."""
    resolved_tenant = (tenant_id or env.profile_id).strip()
    if not resolved_tenant:
        raise ValueError("production capacity tenant authority requires profile_id")
    principal = RequestIdentity(
        tenant_id=resolved_tenant,
        user_id=service_id,
        principal_type=PrincipalType.SERVICE,
        auth_subject=service_id,
    )
    return ProductionCapacityGovernance(
        principal=principal,
        mutation_authorization_boundary=mutation_authorization_boundary,
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=resolved_tenant),
        tenant_id=resolved_tenant,
    )
