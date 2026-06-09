# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for dispute_sim_application (Phase H-APP.5.5, DX-5.5)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.catalog_manifests import OTEL
from dispute_sim_application.host.settings import DisputeSimBackendSettings


def build_dispute_sim_environment_profile(
    settings: DisputeSimBackendSettings,
) -> ApplicationEnvironmentProfile:
    _ = settings
    profile = ApplicationEnvironmentProfile.product_defaults(
        skill_bundles=["harness"],
        profile_id="dispute_sim.product",
    )
    profile.observability_profile.otel_enabled = True
    profile.observability_profile.debug_surface_override = True
    otel_backend = IntegrationBinding.from_manifest(OTEL)
    profile.integration_profile = profile.integration_profile.model_copy(
        update={
            "observability_backend": otel_backend,
            "options": {**profile.integration_profile.options, OTEL.slug: {}},
        },
    )
    return profile
