# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Confluence integration bundle — the single composition root for Confluence in Intergrax.

HTTP clients are opened only in ``opens.py``. Tier-3 code MUST use
``create_confluence_wiki_knowledge()``, ``create_confluence_integration()``, or
``profile.resolve(IntegrationCategory.WIKI_KNOWLEDGE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.adapter import ConfluenceWikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.client import ConfluenceRestClient
from intergrax.integrations.providers.wiki_knowledge.confluence.config import ConfluenceIntegrationConfig
from intergrax.integrations.providers.wiki_knowledge.confluence.opens import (
    open_confluence_rest_client,
    open_confluence_wiki_knowledge,
)


@dataclass(frozen=True)
class ConfluenceIntegrationBundle:
    config: ConfluenceIntegrationConfig
    wiki_knowledge: ConfluenceWikiKnowledge
    rest_client: ConfluenceRestClient


def resolve_confluence_config(**overrides: object) -> ConfluenceIntegrationConfig:
    return ConfluenceIntegrationConfig.from_env(**overrides)


def create_confluence_integration(
    *,
    wiki_knowledge: Optional[WikiKnowledge] = None,
    client: Optional[ConfluenceRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ConfluenceIntegrationConfig], Any]] = None,
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
    assert isinstance(wiki, ConfluenceWikiKnowledge)
    return ConfluenceIntegrationBundle(config=config, wiki_knowledge=wiki, rest_client=rest_client)


def create_confluence_wiki_knowledge(
    *,
    wiki_knowledge: Optional[WikiKnowledge] = None,
    client: Optional[ConfluenceRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ConfluenceIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> ConfluenceWikiKnowledge:
    """Catalog factory for ``IntegrationSlug.CONFLUENCE`` / ``WIKI_KNOWLEDGE``."""
    return create_confluence_integration(
        wiki_knowledge=wiki_knowledge,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).wiki_knowledge
