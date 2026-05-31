# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register brave."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.brave.bundle import create_brave_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_brave_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.BRAVE.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_brave_search_provider,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_BRAVE",
            description="brave integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
