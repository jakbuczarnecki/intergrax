# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_sharepoint_wiki_knowledge as _legacy_create_sharepoint_wiki_knowledge

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.wiki_knowledge.sharepoint.integration import (
    SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID,
    SharepointWikiKnowledgeIntegration,
    SharepointWikiKnowledgeIntegrationConfig,
    SharepointWikiKnowledgeClient,
)

__all__ = [
    "create_sharepoint_wiki_knowledge",
    "create_sharepoint_wiki_knowledge_integration",
]


def create_sharepoint_wiki_knowledge_integration(
    *,
    client: SharepointWikiKnowledgeClient | None = None,
    enabled: bool = False,
) -> SharepointWikiKnowledgeIntegration:
    """
    Build a contract-based Sharepoint wiki knowledge integration.

    The legacy facade (create_sharepoint_wiki_knowledge) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Sharepoint wiki knowledge integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SharepointWikiKnowledgeIntegration.from_client(client, enabled=enabled)
    return SharepointWikiKnowledgeIntegration.for_provider(
        provider_id=SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID,
        display_name="Sharepoint",
        config=SharepointWikiKnowledgeIntegrationConfig(enabled=enabled),
    )


def create_sharepoint_wiki_knowledge(**kwargs: object) -> SharepointWikiKnowledgeIntegration:
    """Compatibility shim — constructs SharepointWikiKnowledgeIntegration from legacy runtime."""
    runtime = _legacy_create_sharepoint_wiki_knowledge(**kwargs)
    if isinstance(runtime, SharepointWikiKnowledgeIntegration):
        return runtime
    return SharepointWikiKnowledgeIntegration.from_runtime(runtime)
