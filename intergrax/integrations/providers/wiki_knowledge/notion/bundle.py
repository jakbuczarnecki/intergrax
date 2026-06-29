# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_notion_wiki_knowledge as _legacy_create_notion_wiki_knowledge

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.wiki_knowledge.notion.integration import (
    NOTION_WIKI_KNOWLEDGE_PROVIDER_ID,
    NotionWikiKnowledgeIntegration,
    NotionWikiKnowledgeIntegrationConfig,
    NotionWikiKnowledgeClient,
)

__all__ = [
    "create_notion_wiki_knowledge",
    "create_notion_wiki_knowledge_integration",
]


def create_notion_wiki_knowledge_integration(
    *,
    client: NotionWikiKnowledgeClient | None = None,
    enabled: bool = False,
) -> NotionWikiKnowledgeIntegration:
    """
    Build a contract-based Notion wiki knowledge integration.

    The legacy facade (create_notion_wiki_knowledge) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Notion wiki knowledge integration requires an injected client when enabled=True",
        )
    if client is not None:
        return NotionWikiKnowledgeIntegration.from_client(client, enabled=enabled)
    return NotionWikiKnowledgeIntegration.for_provider(
        provider_id=NOTION_WIKI_KNOWLEDGE_PROVIDER_ID,
        display_name="Notion",
        config=NotionWikiKnowledgeIntegrationConfig(enabled=enabled),
    )


def create_notion_wiki_knowledge(**kwargs: object) -> NotionWikiKnowledgeIntegration:
    """Compatibility shim — constructs NotionWikiKnowledgeIntegration from legacy runtime."""
    runtime = _legacy_create_notion_wiki_knowledge(**kwargs)
    if isinstance(runtime, NotionWikiKnowledgeIntegration):
        return runtime
    return NotionWikiKnowledgeIntegration.from_runtime(runtime)
