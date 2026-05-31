# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register tavily."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_tavily_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.TAVILY.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_tavily_search_provider,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_TAVILY",
            description="tavily integration (Phase M.7)",
        ),
        override=override,
    )
