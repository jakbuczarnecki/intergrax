# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_arxiv_search_provider as _legacy_create_arxiv_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.arxiv.integration import (
    ARXIV_SEARCH_PROVIDER_PROVIDER_ID,
    ArxivSearchProviderIntegration,
    ArxivSearchProviderIntegrationConfig,
    ArxivSearchProviderClient,
)

__all__ = [
    "create_arxiv_search_provider",
    "create_arxiv_search_provider_integration",
]


def create_arxiv_search_provider_integration(
    *,
    client: ArxivSearchProviderClient | None = None,
    enabled: bool = False,
) -> ArxivSearchProviderIntegration:
    """
    Build a contract-based Arxiv search provider integration.

    The legacy facade (create_arxiv_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Arxiv search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ArxivSearchProviderIntegration.from_client(client, enabled=enabled)
    return ArxivSearchProviderIntegration.for_provider(
        provider_id=ARXIV_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Arxiv",
        config=ArxivSearchProviderIntegrationConfig(enabled=enabled),
    )


def create_arxiv_search_provider(**kwargs: object) -> ArxivSearchProviderIntegration:
    """Compatibility shim — constructs ArxivSearchProviderIntegration from legacy runtime."""
    runtime = _legacy_create_arxiv_search_provider(**kwargs)
    if isinstance(runtime, ArxivSearchProviderIntegration):
        return runtime
    return ArxivSearchProviderIntegration.from_client(runtime)
