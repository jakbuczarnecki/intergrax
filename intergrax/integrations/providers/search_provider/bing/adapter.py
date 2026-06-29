# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bing Web Search adapter — wraps ``websearch.providers.bing_provider``."""

from __future__ import annotations

from typing import Sequence

from intergrax.websearch.providers.bing_provider import BingWebProvider
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit


class _BingSearchProvider:
    """
    Integration-catalog facade over ``BingWebProvider``.

    Implements ``SearchProvider``; exposes ``web_search_provider`` for Tier-0 pipelines
    that still use ``QuerySpec`` (e.g. ``WebSearchExecutor``).
    """

    def __init__(self, provider: BingWebProvider) -> None:
        self._provider = provider

    @property
    def web_search_provider(self) -> BingWebProvider:
        return self._provider

    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        spec = QuerySpec(query=query, top_k=max(1, int(limit)))
        return self._provider.search(spec)
