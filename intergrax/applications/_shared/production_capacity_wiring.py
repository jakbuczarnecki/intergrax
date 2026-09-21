# © Artur Czarnecki. All rights reserved.

"""Production Celery/K8s capacity adapter wiring (AUDIT-IDEAL-30.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.production_capacity_governance_wiring import (
    ProductionCapacityGovernance,
    build_production_capacity_governance,
)
from intergrax.applications._shared.control_plane_composition import (
    product_consequential_capacity_mutations_enabled,
    require_control_plane_mutation_boundary,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.capacity.control_plane_governance import EcpGovernanceBlockedError
from intergrax.runtime.capacity.production_adapters import (
    ProductionCapacityAdapters,
    apply_production_scale_probe,
    build_production_capacity_adapters,
)


@dataclass(frozen=True, slots=True)
class ProductionCapacityWiring:
    enabled: bool
    adapters: ProductionCapacityAdapters | None
    probe_passed: bool


def resolve_production_capacity_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    governance: ProductionCapacityGovernance | None = None,
) -> ProductionCapacityWiring:
    """Enable production-scale Celery/K8s adapters on product hosts."""
    scaling = env.scaling_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ProductionCapacityWiring(enabled=False, adapters=None, probe_passed=False)
    if not scaling.production_adapters_enabled:
        return ProductionCapacityWiring(enabled=False, adapters=None, probe_passed=False)

    resolved_governance = governance or build_production_capacity_governance(env)
    if product_consequential_capacity_mutations_enabled(env):
        require_control_plane_mutation_boundary(
            resolved_governance.mutation_authorization_boundary,
            blocker_code="ECP_BLOCKED_MISSING_BOUNDARY",
            message=(
                "production capacity wiring requires "
                "ControlPlaneMutationAuthorizationBoundary"
            ),
        )
    elif resolved_governance.mutation_authorization_boundary is None:
        return ProductionCapacityWiring(enabled=True, adapters=None, probe_passed=False)
    adapters = build_production_capacity_adapters(
        mutation_boundary=resolved_governance.mutation_authorization_boundary,
        tenant_resolver=resolved_governance.tenant_resolver,
    )
    try:
        probe_passed = apply_production_scale_probe(
            adapters,
            principal=resolved_governance.principal,
            tenant_id=resolved_governance.tenant_id,
            k8s_mutation_id="probe-k8s-scale",
            celery_mutation_id="probe-celery-scale",
        )
    except EcpGovernanceBlockedError:
        probe_passed = False
    return ProductionCapacityWiring(enabled=True, adapters=adapters, probe_passed=probe_passed)
