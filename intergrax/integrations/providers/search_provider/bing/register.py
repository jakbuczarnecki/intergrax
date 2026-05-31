# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Bing in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.bing.bundle import create_bing_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_bing_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.BING.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_bing_search_provider,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_BING",
            description=(
                "Bing Web Search v7 — web search via REST API "
                "(via create_bing_integration; legacy BING_SEARCH_V7_API_KEY supported)"
            ),
        ),
        override=override,
    )
