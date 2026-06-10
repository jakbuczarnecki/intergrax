# © Artur Czarnecki. All rights reserved.

"""Integration marketplace catalog wiring (AUDIT-IDEAL-13.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.marketplace_catalog import (
    IntegrationMarketplaceCatalog,
    build_integration_marketplace_catalog,
)


@dataclass(frozen=True, slots=True)
class IntegrationMarketplaceWiring:
    enabled: bool
    catalog: IntegrationMarketplaceCatalog | None


def resolve_integration_marketplace_wiring(
    env: ApplicationEnvironmentProfile,
) -> IntegrationMarketplaceWiring:
    """Expose trust-scored integration marketplace catalog on product hosts."""
    governance = env.integration_governance_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return IntegrationMarketplaceWiring(enabled=False, catalog=None)
    if not governance.marketplace_catalog_enabled:
        return IntegrationMarketplaceWiring(enabled=False, catalog=None)

    register_default_integrations(preset="core")
    catalog = build_integration_marketplace_catalog()
    if not catalog.entries:
        return IntegrationMarketplaceWiring(enabled=False, catalog=None)
    return IntegrationMarketplaceWiring(enabled=True, catalog=catalog)
