# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_wikipedia_wiki_knowledge as _legacy_create_wikipedia_wiki_knowledge

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.wiki_knowledge.wikipedia.integration import (
    WIKIPEDIA_WIKI_KNOWLEDGE_PROVIDER_ID,
    WikipediaWikiKnowledgeIntegration,
    WikipediaWikiKnowledgeIntegrationConfig,
    WikipediaWikiKnowledgeClient,
)

__all__ = [
    "create_wikipedia_wiki_knowledge",
    "create_wikipedia_wiki_knowledge_integration",
]


def create_wikipedia_wiki_knowledge_integration(
    *,
    client: WikipediaWikiKnowledgeClient | None = None,
    enabled: bool = False,
) -> WikipediaWikiKnowledgeIntegration:
    """
    Build a contract-based Wikipedia wiki knowledge integration.

    The legacy facade (create_wikipedia_wiki_knowledge) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Wikipedia wiki knowledge integration requires an injected client when enabled=True",
        )
    if client is not None:
        return WikipediaWikiKnowledgeIntegration.from_client(client, enabled=enabled)
    return WikipediaWikiKnowledgeIntegration.for_provider(
        provider_id=WIKIPEDIA_WIKI_KNOWLEDGE_PROVIDER_ID,
        display_name="Wikipedia",
        config=WikipediaWikiKnowledgeIntegrationConfig(enabled=enabled),
    )


def create_wikipedia_wiki_knowledge(**kwargs: object) -> WikipediaWikiKnowledgeIntegration:
    """Compatibility shim — constructs WikipediaWikiKnowledgeIntegration from legacy runtime."""
    runtime = _legacy_create_wikipedia_wiki_knowledge(**kwargs)
    if isinstance(runtime, WikipediaWikiKnowledgeIntegration):
        return runtime
    return WikipediaWikiKnowledgeIntegration.from_client(runtime)
