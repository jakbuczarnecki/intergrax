# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki adapter — ``WikiKnowledge`` facade (no HTTP here)."""

from __future__ import annotations

from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.integrations.providers.wiki_knowledge.confluence.client import ConfluenceRestClient


class ConfluenceWikiKnowledge:
    """
    Catalog facade over ``ConfluenceRestClient``.

    Instantiate via ``create_confluence_wiki_knowledge()`` — not from agent code.
    """

    def __init__(self, client: ConfluenceRestClient) -> None:
        self._client = client

    @property
    def rest_client(self) -> ConfluenceRestClient:
        return self._client

    def get_page(self, page_id: str) -> WikiPageRecord:
        return self._client.get_page(page_id)

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        return self._client.search_pages(query, limit=limit)
