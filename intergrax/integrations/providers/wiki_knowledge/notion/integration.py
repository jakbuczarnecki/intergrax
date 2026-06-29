# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Notion wiki knowledge integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
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
    Single public Notion wiki knowledge entrypoint.

    Legacy catalog factory (create_notion_wiki_knowledge) delegates to this class.
    """

    config: NotionWikiKnowledgeIntegrationConfig = NotionWikiKnowledgeIntegrationConfig()
    _client: NotionWikiKnowledgeClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> NotionWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=NOTION_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Notion",
            config=NotionWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Notion integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

WikiKnowledge.register(NotionWikiKnowledgeIntegration)
