# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "NOTION_WIKI_KNOWLEDGE_PROVIDER_ID",
    "NotionWikiKnowledgeIntegration",
    "NotionWikiKnowledgeIntegrationConfig",
    "NotionWikiKnowledgeClient",
    "create_notion_wiki_knowledge",
    "create_notion_wiki_knowledge_integration",
    "register_notion_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_notion_wiki_knowledge",
        "create_notion_wiki_knowledge_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "NOTION_WIKI_KNOWLEDGE_PROVIDER_ID",
        "NotionWikiKnowledgeIntegration",
        "NotionWikiKnowledgeIntegrationConfig",
        "NotionWikiKnowledgeClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "NOTION_WIKI_KNOWLEDGE_PROVIDER_ID",
        "NotionWikiKnowledgeIntegration",
        "NotionWikiKnowledgeIntegrationConfig",
        "NotionWikiKnowledgeClient",
    }
)

def __getattr__(name: str):
    if name == "register_notion_integration":
        from intergrax.integrations.providers.wiki_knowledge.notion.register import register_notion_integration

        return register_notion_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.notion import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.notion import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.notion import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
