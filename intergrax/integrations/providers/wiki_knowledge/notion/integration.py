# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Notion wiki knowledge integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NOTION_WIKI_KNOWLEDGE_PROVIDER_ID = "notion"


class NotionWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Notion wiki knowledge integration."""

    pass


@runtime_checkable
class NotionWikiKnowledgeClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class NotionWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Notion wiki knowledge integration.

    The legacy facade (create_notion_wiki_knowledge) remains separate and backward-compatible.
    """

    config: NotionWikiKnowledgeIntegrationConfig = NotionWikiKnowledgeIntegrationConfig()
    _client: NotionWikiKnowledgeClient | None = PrivateAttr(default=None)

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
