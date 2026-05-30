# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Confluence openers — internal to the confluence integration package.

Only this module may construct ``httpx.Client`` / ``ConfluenceRestClient`` for Confluence.
All composition roots use ``bundle.create_confluence_*`` or ``profile.resolve(WIKI_KNOWLEDGE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.providers.confluence.adapter import ConfluenceWikiKnowledge
from intergrax.integrations.providers.confluence.client import ConfluenceRestClient
from intergrax.integrations.providers.confluence.config import (
    DEFAULT_TIMEOUT_SECONDS,
    ConfluenceIntegrationConfig,
)


def _create_http_client(config: ConfluenceIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    return httpx.Client(
        base_url=config.api_base_url,
        auth=(config.email, config.api_token),
        timeout=timeout,
        headers={"Accept": "application/json"},
    )


def open_confluence_rest_client(
    config: ConfluenceIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ConfluenceIntegrationConfig], Any]] = None,
) -> ConfluenceRestClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return ConfluenceRestClient(config, http_client=http_client)


def open_confluence_wiki_knowledge(
    config: ConfluenceIntegrationConfig,
    *,
    implementation: Optional[WikiKnowledge] = None,
    client: Optional[ConfluenceRestClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[ConfluenceIntegrationConfig], Any]] = None,
) -> WikiKnowledge:
    if implementation is not None:
        return implementation
    rest_client = client or open_confluence_rest_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return ConfluenceWikiKnowledge(rest_client)
