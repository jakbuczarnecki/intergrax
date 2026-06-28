# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_semantic_scholar_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.semantic_scholar.integration import (
    SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID,
    SemanticScholarSearchProviderIntegration,
    SemanticScholarSearchProviderIntegrationConfig,
    SemanticScholarSearchProviderClient,
)

__all__ = [
    "create_semantic_scholar_search_provider",
    "create_semantic_scholar_search_provider_integration",
]


def create_semantic_scholar_search_provider_integration(
    *,
    client: SemanticScholarSearchProviderClient | None = None,
    enabled: bool = False,
) -> SemanticScholarSearchProviderIntegration:
    """
    Build a contract-based Semantic Scholar search provider integration.

    The legacy facade (create_semantic_scholar_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Semantic Scholar search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SemanticScholarSearchProviderIntegration.from_client(client, enabled=enabled)
    return SemanticScholarSearchProviderIntegration.for_provider(
        provider_id=SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Semantic Scholar",
        config=SemanticScholarSearchProviderIntegrationConfig(enabled=enabled),
    )
