# © Artur Czarnecki. All rights reserved.

"""Integration catalog hot-reload wiring (AUDIT-IDEAL-13.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.catalog_hot_reload import (
    CatalogHotReloadReport,
    reload_integration_catalog,
)


@dataclass(frozen=True, slots=True)
class CatalogHotReloadWiring:
    enabled: bool
    report: CatalogHotReloadReport | None


def resolve_catalog_hot_reload_wiring(
    env: ApplicationEnvironmentProfile,
) -> CatalogHotReloadWiring:
    """Enable in-process integration catalog hot-reload on product hosts."""
    governance = env.integration_governance_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CatalogHotReloadWiring(enabled=False, report=None)
    if not governance.catalog_hot_reload_enabled:
        return CatalogHotReloadWiring(enabled=False, report=None)

    report = reload_integration_catalog(preset="core")
    if not report.reloaded:
        return CatalogHotReloadWiring(enabled=False, report=None)
    return CatalogHotReloadWiring(enabled=True, report=report)
