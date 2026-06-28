# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Wikipedia wiki knowledge integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WIKIPEDIA_WIKI_KNOWLEDGE_PROVIDER_ID = "wikipedia"


class WikipediaWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Wikipedia wiki knowledge integration."""

    pass


@runtime_checkable
class WikipediaWikiKnowledgeClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class WikipediaWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Wikipedia wiki knowledge integration.

    The legacy facade (create_wikipedia_wiki_knowledge) remains separate and backward-compatible.
    """

    config: WikipediaWikiKnowledgeIntegrationConfig = WikipediaWikiKnowledgeIntegrationConfig()
    _client: WikipediaWikiKnowledgeClient | None = PrivateAttr(default=None)

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
