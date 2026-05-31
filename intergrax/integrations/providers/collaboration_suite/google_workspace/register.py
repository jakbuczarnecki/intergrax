# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register google_workspace."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import create_google_workspace_collaboration_suite
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_google_workspace_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GOOGLE_WORKSPACE.value,
            categories=(IntegrationCategory.COLLABORATION_SUITE,),
            factory=create_google_workspace_collaboration_suite,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_GOOGLE_WORKSPACE",
            description="google_workspace integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
