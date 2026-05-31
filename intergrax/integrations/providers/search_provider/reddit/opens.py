# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.integrations.providers.search_provider.reddit.config import RedditIntegrationConfig
from intergrax.integrations.providers.search_provider.reddit.web_client import RedditAPIProvider
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit


class RedditSearchProvider:
    def __init__(self, provider: RedditAPIProvider) -> None:
        self._provider = provider

    @property
    def web_search_provider(self) -> RedditAPIProvider:
        return self._provider

    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        spec = QuerySpec(query=query, top_k=max(1, int(limit)))
        return self._provider.search(spec)


def open_reddit_web_search_provider(
    config: RedditIntegrationConfig,
    *,
    provider: Optional[RedditAPIProvider] = None,
    session: Optional[object] = None,
) -> RedditAPIProvider:
    if provider is not None:
        return provider
    timeout = int(config.timeout_seconds or RedditAPIProvider._DEFAULT_TIMEOUT)
    return RedditAPIProvider(
        client_id=config.client_id,
        client_secret=config.client_secret,
        user_agent=config.user_agent,
        session=session,
        timeout=timeout,
        include_comments=config.include_comments,
        comments_limit=config.comments_limit,
    )


def open_reddit_search_provider(
    config: RedditIntegrationConfig,
    *,
    provider: Optional[RedditAPIProvider] = None,
    session: Optional[object] = None,
) -> RedditSearchProvider:
    return RedditSearchProvider(
        open_reddit_web_search_provider(config, provider=provider, session=session)
    )
