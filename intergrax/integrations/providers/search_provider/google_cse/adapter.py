# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google CSE search adapter — wraps ``websearch.providers.google_cse_provider``."""

from __future__ import annotations

from typing import Sequence

from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit


class _GoogleCSESearchProvider:
    """
    Integration-catalog facade over ``GoogleCSEProvider``.

    Implements ``SearchProvider``; exposes ``web_search_provider`` for Tier-0 pipelines
    that still use ``QuerySpec`` (e.g. ``WebSearchExecutor``).
    """

    def __init__(self, provider: GoogleCSEProvider) -> None:
        self._provider = provider

    @property
    def web_search_provider(self) -> GoogleCSEProvider:
        return self._provider

    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        spec = QuerySpec(query=query, top_k=max(1, int(limit)))
        return self._provider.search(spec)
