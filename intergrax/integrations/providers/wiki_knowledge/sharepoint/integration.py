# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sharepoint wiki knowledge integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
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
    Single public Sharepoint wiki knowledge entrypoint.

    Legacy catalog factory (create_sharepoint_wiki_knowledge) delegates to this class.
    """

    config: SharepointWikiKnowledgeIntegrationConfig = SharepointWikiKnowledgeIntegrationConfig()
    _client: SharepointWikiKnowledgeClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> SharepointWikiKnowledgeIntegration:
        integration = cls.for_provider(
            provider_id=SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID,
            display_name="Sharepoint",
            config=SharepointWikiKnowledgeIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Sharepoint integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

WikiKnowledge.register(SharepointWikiKnowledgeIntegration)
