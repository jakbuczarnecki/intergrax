# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki integration (Phase M.6)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.wiki_knowledge.confluence.config import (
    ENV_CONFLUENCE_API_TOKEN,
    ENV_CONFLUENCE_BASE_URL,
    ENV_CONFLUENCE_EMAIL,
    ConfluenceIntegrationConfig,
)

__all__ = [
    "ENV_CONFLUENCE_API_TOKEN",
    "ENV_CONFLUENCE_BASE_URL",
    "ENV_CONFLUENCE_EMAIL",
    "ConfluenceIntegrationBundle",
    "ConfluenceIntegrationConfig",
    "ConfluenceWikiKnowledge",
    "create_confluence_integration",
    "create_confluence_wiki_knowledge",
    "register_confluence_integration",
    "resolve_confluence_config",
    "create_confluence_wiki_knowledge_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "ConfluenceIntegrationBundle",
        "ConfluenceWikiKnowledge",
        "create_confluence_integration",
        "create_confluence_wiki_knowledge",
        "register_confluence_integration",
        "resolve_confluence_config",
        "create_confluence_wiki_knowledge_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID",
        "ConfluenceWikiKnowledgeIntegration",
        "ConfluenceWikiKnowledgeIntegrationConfig",
        "ConfluenceWikiKnowledgeClient",
    }
)

def __getattr__(name: str):
    if name == "register_confluence_integration":
        from intergrax.integrations.providers.wiki_knowledge.confluence.register import register_confluence_integration

        return register_confluence_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.confluence import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "ConfluenceWikiKnowledge":
        from intergrax.integrations.providers.wiki_knowledge.confluence.adapter import ConfluenceWikiKnowledge

        return ConfluenceWikiKnowledge
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.confluence import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
