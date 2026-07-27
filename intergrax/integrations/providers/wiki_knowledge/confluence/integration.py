# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki knowledge integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    ConfluenceKnowledgePage,
    ConfluenceKnowledgePagePage,
    ConfluenceKnowledgeReadClient,
)
from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID = "confluence"


class ConfluenceWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Confluence wiki knowledge integration."""

    pass


ConfluenceWikiKnowledgeClient = WikiKnowledge

class ConfluenceWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Single public Confluence wiki knowledge entrypoint.

    Legacy catalog factory (create_confluence_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: ConfluenceWikiKnowledgeIntegrationConfig = ConfluenceWikiKnowledgeIntegrationConfig()
    _client: ConfluenceWikiKnowledgeClient | None = PrivateAttr(default=None)
    

    def get_page(self, page_id):
        return self._require_client().get_page(page_id)

    def search_pages(self, query, limit: int = 25):
        return self._require_client().search_pages(query, limit=limit)

    def list_knowledge_pages(
        self,
        *,
        space_id: str,
        cursor: str | None,
        limit: int,
    ) -> ConfluenceKnowledgePagePage:
        return self._require_knowledge_client().list_knowledge_pages(
            space_id=space_id,
            cursor=cursor,
            limit=limit,
        )

    def get_knowledge_page(
        self,
        *,
        page_id: str,
        version_number: int,
    ) -> ConfluenceKnowledgePage:
        return self._require_knowledge_client().get_knowledge_page(
            page_id=page_id,
            version_number=version_number,
        )

    def _require_knowledge_client(self) -> ConfluenceKnowledgeReadClient:
        client = self._require_client()
        if not isinstance(client, ConfluenceKnowledgeReadClient):
            raise IntegrationConfigurationError(
                "Confluence integration does not expose knowledge read capability",
            )
        return client

    def _require_client(self) -> WikiKnowledge:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: ConfluenceWikiKnowledgeClient,
        *,
        enabled: bool = False,
    ) -> ConfluenceWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Confluence",
            config=ConfluenceWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ConfluenceWikiKnowledgeClient | None:
        return self._client

WikiKnowledge.register(ConfluenceWikiKnowledgeIntegration)
