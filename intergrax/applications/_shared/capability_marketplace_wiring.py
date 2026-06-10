# © Artur Czarnecki. All rights reserved.

"""Capability marketplace readiness wiring (AUDIT-IDEAL-AHI.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.marketplace_catalog import build_integration_marketplace_catalog
from intergrax.runtime.adaptive.capability_marketplace_readiness import (
    CapabilityMarketplaceReadinessReport,
    evaluate_capability_marketplace_readiness,
)


@dataclass(frozen=True, slots=True)
class CapabilityMarketplaceWiring:
    enabled: bool
    report: CapabilityMarketplaceReadinessReport | None


def resolve_capability_marketplace_wiring(
    env: ApplicationEnvironmentProfile,
) -> CapabilityMarketplaceWiring:
    """Evaluate marketplace readiness when adaptive profile enables the path."""
    adaptive = env.adaptive_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CapabilityMarketplaceWiring(enabled=False, report=None)
    if not adaptive.enabled or not adaptive.capability_marketplace_enabled:
        return CapabilityMarketplaceWiring(enabled=False, report=None)

    register_default_integrations(preset="full", override=True)
    marketplace_catalog = build_integration_marketplace_catalog()
    if not marketplace_catalog.entries:
        return CapabilityMarketplaceWiring(enabled=False, report=None)

    report = evaluate_capability_marketplace_readiness(marketplace_catalog=marketplace_catalog)
    if not report.ready:
        return CapabilityMarketplaceWiring(enabled=False, report=report)
    return CapabilityMarketplaceWiring(enabled=True, report=report)
