# © Artur Czarnecki. All rights reserved.

"""Production Celery/K8s capacity adapter wiring (AUDIT-IDEAL-30.4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.applications._shared.control_plane_composition import (
    product_consequential_capacity_mutations_enabled,
    require_control_plane_mutation_boundary,
)
from intergrax.applications._shared.production_capacity_governance_wiring import (
    ProductionCapacityGovernance,
    build_production_capacity_governance,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.capacity.control_plane_governance import EcpGovernanceBlockedError
from intergrax.runtime.capacity.production_adapters import (
    CapacityBackendNotConfiguredError,
    CapacityBackendPosture,
    ProductionCapacityAdapters,
    apply_production_scale_probe,
    build_production_capacity_adapters,
)


class CapacityQualification(StrEnum):
    """What the capacity wiring can honestly claim (T-IGX-043-B, DP-29).

    There is deliberately no fully-qualified member: no live Celery scaler exists, and live
    capacity qualification stays a PROD-Q matter.
    """

    NOT_ENABLED = "not_enabled"
    UNQUALIFIED_NO_CLUSTER = "unqualified_no_cluster"
    CI_DETERMINISTIC_ONLY = "ci_deterministic_only"
    LIVE_KUBERNETES_CELERY_INTENT_ONLY = "live_kubernetes_celery_intent_only"


@dataclass(frozen=True, slots=True)
class ProductionCapacityWiring:
    enabled: bool
    adapters: ProductionCapacityAdapters | None
    probe_passed: bool
    qualification: CapacityQualification = CapacityQualification.NOT_ENABLED
    reason: str = ""


def resolve_production_capacity_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    governance: ProductionCapacityGovernance | None = None,
    backend_posture: CapacityBackendPosture = CapacityBackendPosture.PRODUCTION,
) -> ProductionCapacityWiring:
    """Enable production-scale Celery/K8s adapters on product hosts.

    Without a configured cluster the wiring is reported ``UNQUALIFIED_NO_CLUSTER`` and builds
    no adapters, so capacity mutations are refused (DP-29).
    """
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
            message=("production capacity wiring requires ControlPlaneMutationAuthorizationBoundary"),
        )
    elif resolved_governance.mutation_authorization_boundary is None:
        return ProductionCapacityWiring(
            enabled=True,
            adapters=None,
            probe_passed=False,
            reason="mutation authorization boundary not supplied",
        )
    try:
        adapters = build_production_capacity_adapters(
            mutation_boundary=resolved_governance.mutation_authorization_boundary,
            tenant_resolver=resolved_governance.tenant_resolver,
            backend_posture=backend_posture,
        )
    except CapacityBackendNotConfiguredError as exc:
        return ProductionCapacityWiring(
            enabled=True,
            adapters=None,
            probe_passed=False,
            qualification=CapacityQualification.UNQUALIFIED_NO_CLUSTER,
            reason=str(exc),
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
    live = adapters.kubernetes_backend == "live"
    return ProductionCapacityWiring(
        enabled=True,
        adapters=adapters,
        probe_passed=probe_passed and live,
        qualification=(
            CapacityQualification.LIVE_KUBERNETES_CELERY_INTENT_ONLY
            if live
            else CapacityQualification.CI_DETERMINISTIC_ONLY
        ),
        reason="" if live else "in-memory scaler (explicit CI posture)",
    )
