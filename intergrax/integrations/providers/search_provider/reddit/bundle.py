# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.reddit.config import RedditIntegrationConfig
from intergrax.integrations.providers.search_provider.reddit.opens import open_reddit_search_provider


def create_reddit_search_provider(**config_overrides: object) -> RedditSearchProviderIntegration:
    config = RedditIntegrationConfig.from_env(**config_overrides)
    return RedditSearchProviderIntegration.from_client(open_reddit_search_provider(config))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.reddit.integration import (
    REDDIT_SEARCH_PROVIDER_PROVIDER_ID,
    RedditSearchProviderIntegration,
    RedditSearchProviderIntegrationConfig,
    RedditSearchProviderClient,
)


def create_reddit_search_provider_integration(
    *,
    client: RedditSearchProviderClient | None = None,
    enabled: bool = False,
) -> RedditSearchProviderIntegration:
    """
    Build a contract-based Reddit search provider integration.

    The legacy facade (create_reddit_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Reddit search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return RedditSearchProviderIntegration.from_client(client, enabled=enabled)
    return RedditSearchProviderIntegration.for_provider(
        provider_id=REDDIT_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Reddit",
        config=RedditSearchProviderIntegrationConfig(enabled=enabled),
    )
