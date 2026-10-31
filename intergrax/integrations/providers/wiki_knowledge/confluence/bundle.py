# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Confluence integration bundle — the single composition root for Confluence in Intergrax.

HTTP clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_confluence_wiki_knowledge()``, ``create_confluence_integration()``, or
``profile.resolve(IntegrationCategory.WIKI_KNOWLEDGE)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.client import ConfluenceRestClient
from intergrax.integrations.providers.wiki_knowledge.confluence.config import ConfluenceIntegrationConfig
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
    ConfluenceWikiKnowledgeClient,
    ConfluenceWikiKnowledgeIntegration,
    ConfluenceWikiKnowledgeIntegrationConfig,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.opens import (
    open_confluence_rest_client,
    open_confluence_wiki_knowledge,
)


@dataclass(frozen=True)
class ConfluenceIntegrationBundle:
    config: ConfluenceIntegrationConfig
    wiki_knowledge: ConfluenceWikiKnowledgeIntegration
    rest_client: ConfluenceRestClient


def resolve_confluence_config(**overrides: object) -> ConfluenceIntegrationConfig:
    return ConfluenceIntegrationConfig.from_env(**overrides)


def create_confluence_integration(
    *,
    wiki_knowledge: WikiKnowledge | None = None,
    client: ConfluenceRestClient | None = None,
    http_client: Any | None = None,
    http_client_factory: Callable[[ConfluenceIntegrationConfig], Any] | None = None,
    **config_overrides: object,
) -> ConfluenceIntegrationBundle:
    config = resolve_confluence_config(**config_overrides)
    rest_client = client or open_confluence_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    wiki = open_confluence_wiki_knowledge(
        config,
        implementation=wiki_knowledge,
        client=rest_client,
    )
    assert isinstance(wiki, ConfluenceWikiKnowledgeIntegration)
    return ConfluenceIntegrationBundle(config=config, wiki_knowledge=wiki, rest_client=rest_client)


def create_confluence_wiki_knowledge_integration(
    *,
    client: ConfluenceWikiKnowledgeClient | None = None,
    enabled: bool = False,
) -> ConfluenceWikiKnowledgeIntegration:
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Confluence wiki knowledge integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ConfluenceWikiKnowledgeIntegration.from_client(client, enabled=enabled)
    return ConfluenceWikiKnowledgeIntegration.for_provider(
        provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
        display_name="Confluence",
        config=ConfluenceWikiKnowledgeIntegrationConfig(enabled=enabled),
    )


def create_confluence_wiki_knowledge(
    *,
    wiki_knowledge: WikiKnowledge | None = None,
    client: ConfluenceRestClient | None = None,
    http_client: Any | None = None,
    http_client_factory: Callable[[ConfluenceIntegrationConfig], Any] | None = None,
    **config_overrides: object,
) -> ConfluenceWikiKnowledgeIntegration:
    """Catalog factory for ``"confluence"`` / ``WIKI_KNOWLEDGE``."""
    if (
        wiki_knowledge is None
        and client is None
        and http_client is None
        and http_client_factory is None
        and not config_overrides
    ):
        return create_confluence_wiki_knowledge_integration(enabled=False)
    return create_confluence_integration(
        wiki_knowledge=wiki_knowledge,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).wiki_knowledge
