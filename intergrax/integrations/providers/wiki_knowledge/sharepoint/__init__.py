# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID",
    "SharepointWikiKnowledgeIntegration",
    "SharepointWikiKnowledgeIntegrationConfig",
    "SharepointWikiKnowledgeClient",
    "create_sharepoint_wiki_knowledge",
    "create_sharepoint_wiki_knowledge_integration",
    "register_sharepoint_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_sharepoint_wiki_knowledge",
        "create_sharepoint_wiki_knowledge_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID",
        "SharepointWikiKnowledgeIntegration",
        "SharepointWikiKnowledgeIntegrationConfig",
        "SharepointWikiKnowledgeClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID",
        "SharepointWikiKnowledgeIntegration",
        "SharepointWikiKnowledgeIntegrationConfig",
        "SharepointWikiKnowledgeClient",
    }
)

def __getattr__(name: str):
    if name == "register_sharepoint_integration":
        from intergrax.integrations.providers.wiki_knowledge.sharepoint.register import register_sharepoint_integration

        return register_sharepoint_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.sharepoint import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.sharepoint import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.wiki_knowledge.sharepoint import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
