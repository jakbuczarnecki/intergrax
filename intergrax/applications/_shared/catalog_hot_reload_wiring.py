# © Artur Czarnecki. All rights reserved.

"""Integration catalog hot-reload wiring (AUDIT-IDEAL-13.2, GR-12-A4-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.catalog_hot_reload_service import CatalogHotReloadService
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


@dataclass(frozen=True, slots=True)
class CatalogHotReloadWiring:
    enabled: bool
    service: CatalogHotReloadService | None


def resolve_catalog_hot_reload_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    operator_principal: RequestIdentity | None = None,
) -> CatalogHotReloadWiring:
    """Expose governed catalog hot-reload capability without executing reload."""
    governance = env.integration_governance_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CatalogHotReloadWiring(enabled=False, service=None)
    if not governance.catalog_hot_reload_enabled:
        return CatalogHotReloadWiring(enabled=False, service=None)

    profile_id = env.profile_id.strip()
    principal = operator_principal or RequestIdentity(
        tenant_id=profile_id,
        user_id="integration-catalog-hot-reload",
        principal_type=PrincipalType.SERVICE,
        auth_subject="integration-catalog-hot-reload",
    )
    service = CatalogHotReloadService(
        mutation_authorization_boundary=mutation_authorization_boundary,
        operator_principal=principal,
    )
    return CatalogHotReloadWiring(enabled=True, service=service)
