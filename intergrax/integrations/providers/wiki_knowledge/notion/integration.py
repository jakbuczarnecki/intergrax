# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Notion wiki knowledge integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NOTION_WIKI_KNOWLEDGE_PROVIDER_ID = "notion"


class NotionWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Notion wiki knowledge integration."""

    pass


NotionWikiKnowledgeClient = WikiKnowledge

class NotionWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Single public Notion wiki knowledge entrypoint.

    Legacy catalog factory (create_notion_wiki_knowledge) owns catalog behavior; legacy factories use from_client().
    """

    config: NotionWikiKnowledgeIntegrationConfig = NotionWikiKnowledgeIntegrationConfig()
    _client: NotionWikiKnowledgeClient | None = PrivateAttr(default=None)
    

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
        client: NotionWikiKnowledgeClient,
        *,
        enabled: bool = False,
    ) -> NotionWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=NOTION_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Notion",
            config=NotionWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> NotionWikiKnowledgeClient | None:
        return self._client

WikiKnowledge.register(NotionWikiKnowledgeIntegration)
