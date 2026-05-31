# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.google_places.bundle import create_google_places_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_google_places_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GOOGLE_PLACES.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_google_places_search_provider,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_GOOGLE_PLACES",
            description="Google Places / Business text search",
        ),
        override=override,
    )
