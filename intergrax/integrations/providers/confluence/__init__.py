# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence wiki integration (Phase M.6)."""

from intergrax.integrations.providers.confluence.config import (
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
]

_LAZY_EXPORTS = frozenset(
    {
        "ConfluenceIntegrationBundle",
        "ConfluenceWikiKnowledge",
        "create_confluence_integration",
        "create_confluence_wiki_knowledge",
        "register_confluence_integration",
        "resolve_confluence_config",
    }
)


def __getattr__(name: str):
    if name == "register_confluence_integration":
        from intergrax.integrations.providers.confluence.register import register_confluence_integration

        return register_confluence_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.confluence import bundle as _bundle

        return getattr(_bundle, name)
    if name == "ConfluenceWikiKnowledge":
        from intergrax.integrations.providers.confluence.adapter import ConfluenceWikiKnowledge

        return ConfluenceWikiKnowledge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
