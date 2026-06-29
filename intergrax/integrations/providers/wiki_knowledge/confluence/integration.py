# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki knowledge integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
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
    Single public Confluence wiki knowledge entrypoint.

    Legacy catalog factory (create_confluence_integration) delegates to this class.
    """

    config: ConfluenceWikiKnowledgeIntegrationConfig = ConfluenceWikiKnowledgeIntegrationConfig()
    _client: _ConfluenceWikiKnowledgeClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> ConfluenceWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Confluence",
            config=ConfluenceWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Confluence integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: _ConfluenceWikiKnowledgeClient,
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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

WikiKnowledge.register(ConfluenceWikiKnowledgeIntegration)
