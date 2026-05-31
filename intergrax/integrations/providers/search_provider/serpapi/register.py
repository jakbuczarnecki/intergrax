# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register serpapi."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.serpapi.bundle import create_serpapi_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_serpapi_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SERPAPI.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_serpapi_search_provider,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SERPAPI",
            description="serpapi integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
