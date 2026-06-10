# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for local_workspace_application."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    HostDeploymentProfile,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.catalog_manifests import OTEL
from intergrax.integrations.registry.profile import IntegrationProfile
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings


def build_local_workspace_environment_profile(
    settings: LocalWorkspaceBackendSettings,
) -> ApplicationEnvironmentProfile:
    _ = settings
    profile = (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="local_workspace.product",
            skill_bundles=["harness"],
        )
        .model_copy(
            update={
                "integration_profile": IntegrationProfile.legal_product(),
                "context_profile": ContextProfile(enable_rag=True, enable_websearch=False),
            }
        )
        .with_harness_memory()
    )
    profile.observability_profile.otel_enabled = True
    profile.observability_profile.debug_surface_override = True
    profile.host_deployment_profile = HostDeploymentProfile(
        lkw_hybrid_daemon_enabled=True,
        lkw_daemon_bind_host="127.0.0.1",
        lkw_daemon_port=8020,
    )
    otel_backend = IntegrationBinding.from_manifest(OTEL)
    profile.integration_profile = profile.integration_profile.model_copy(
        update={
            "observability_backend": otel_backend,
            "options": {**profile.integration_profile.options, OTEL.slug: {}},
        },
    )
    return profile
