# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.reddit.bundle import create_reddit_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_reddit_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.REDDIT.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_reddit_search_provider,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_REDDIT",
            description="Reddit OAuth2 search API",
        ),
        override=override,
    )
