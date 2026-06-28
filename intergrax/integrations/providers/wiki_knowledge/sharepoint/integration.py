# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sharepoint wiki knowledge integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import WikiKnowledgeIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID = "sharepoint"


class SharepointWikiKnowledgeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Sharepoint wiki knowledge integration."""

    pass


@runtime_checkable
class SharepointWikiKnowledgeClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SharepointWikiKnowledgeIntegration(WikiKnowledgeIntegrationContract):
    """
    Sharepoint wiki knowledge integration.

    The legacy facade (create_sharepoint_wiki_knowledge) remains separate and backward-compatible.
    """

    config: SharepointWikiKnowledgeIntegrationConfig = SharepointWikiKnowledgeIntegrationConfig()
    _client: SharepointWikiKnowledgeClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SharepointWikiKnowledgeClient,
        *,
        enabled: bool = False,
    ) -> SharepointWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Sharepoint",
            config=SharepointWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SharepointWikiKnowledgeClient | None:
        return self._client
