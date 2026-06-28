# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki knowledge integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID = "confluence"


class ConfluenceWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Confluence wiki knowledge integration."""

    pass


@runtime_checkable
class ConfluenceWikiKnowledgeClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ConfluenceWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Confluence wiki knowledge integration.

    The legacy facade (create_confluence_integration) remains separate and backward-compatible.
    """

    config: ConfluenceWikiKnowledgeIntegrationConfig = ConfluenceWikiKnowledgeIntegrationConfig()
    _client: ConfluenceWikiKnowledgeClient | None = PrivateAttr(default=None)

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
