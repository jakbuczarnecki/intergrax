# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for governed_contractor_application (Phase H-APP.5.5, DX-5.5)."""

from __future__ import annotations

from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.applications._shared.reference_capability_bundle import harness_platform_tool_profile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.catalog_manifests import LOG, OTEL

from governed_contractor_application.host.settings import GovernedContractorBackendSettings


def build_governed_contractor_environment_profile(
    settings: GovernedContractorBackendSettings,
) -> ApplicationEnvironmentProfile:
    bootstrap_application_integration_catalog(integration_preset="full")
    profile = ApplicationEnvironmentProfile.product_defaults(
        skill_bundles=["harness"],
        profile_id="governed_contractor.product",
    )
    profile = profile.model_copy(
        update={
            "capabilities": profile.capabilities.model_copy(
                update={"tools": harness_platform_tool_profile()},
            ),
        },
    )
    profile.observability_profile.otel_enabled = True
    profile.observability_profile.debug_surface_override = True
    otel_backend = IntegrationBinding.from_manifest(OTEL)
    options = {**profile.integration_profile.options, OTEL.slug: {}}
    integration_updates: dict[str, object] = {
        "observability_backend": otel_backend,
        "options": options,
    }
    if profile.integration_profile.notification_channel is None:
        log_backend = IntegrationBinding.from_manifest(LOG)
        integration_updates["notification_channel"] = log_backend
        integration_updates["options"] = {**options, LOG.slug: {}}
    profile.integration_profile = profile.integration_profile.model_copy(
        update=integration_updates,
    )
    if settings.environment == ApiEnvironment.DEV:
        profile = profile.model_copy(
            update={
                "reliability_profile": profile.reliability_profile.model_copy(
                    update={"middleware_hook_timeout_seconds": 2.0},
                ),
            },
        )
    return profile.with_reference_host_platform_defaults()
