# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Wikipedia wiki knowledge integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WIKIPEDIA_WIKI_KNOWLEDGE_PROVIDER_ID = "wikipedia"


class WikipediaWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Wikipedia wiki knowledge integration."""

    pass


WikipediaWikiKnowledgeClient = WikiKnowledge

class WikipediaWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Single public Wikipedia wiki knowledge entrypoint.

    Legacy catalog factory (create_wikipedia_wiki_knowledge) owns catalog behavior; legacy factories use from_client().
    """

    config: WikipediaWikiKnowledgeIntegrationConfig = WikipediaWikiKnowledgeIntegrationConfig()
    _client: WikipediaWikiKnowledgeClient | None = PrivateAttr(default=None)
    

    def get_page(self, page_id):
        return self._require_client().get_page(page_id)

    def search_pages(self, query, limit: int = 25):
        return self._require_client().search_pages(query, limit=limit)

    def _require_client(self) -> WikiKnowledge:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: WikipediaWikiKnowledgeClient,
        *,
        enabled: bool = False,
    ) -> WikipediaWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=WIKIPEDIA_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Wikipedia",
            config=WikipediaWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WikipediaWikiKnowledgeClient | None:
        return self._client

WikiKnowledge.register(WikipediaWikiKnowledgeIntegration)
