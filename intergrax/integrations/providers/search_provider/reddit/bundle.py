# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.reddit.config import RedditIntegrationConfig
from intergrax.integrations.providers.search_provider.reddit.opens import open_reddit_search_provider


def create_reddit_search_provider(**config_overrides: object) -> SearchProvider:
    config = RedditIntegrationConfig.from_env(**config_overrides)
    return open_reddit_search_provider(config)
