# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_exa_search_provider as _legacy_create_exa_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.exa.integration import (
    EXA_SEARCH_PROVIDER_PROVIDER_ID,
    ExaSearchProviderIntegration,
    ExaSearchProviderIntegrationConfig,
    ExaSearchProviderClient,
)

__all__ = [
    "create_exa_search_provider",
    "create_exa_search_provider_integration",
]


def create_exa_search_provider_integration(
    *,
    client: ExaSearchProviderClient | None = None,
    enabled: bool = False,
) -> ExaSearchProviderIntegration:
    """
    Build a contract-based Exa search provider integration.

    The legacy facade (create_exa_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Exa search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ExaSearchProviderIntegration.from_client(client, enabled=enabled)
    return ExaSearchProviderIntegration.for_provider(
        provider_id=EXA_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Exa",
        config=ExaSearchProviderIntegrationConfig(enabled=enabled),
    )


def create_exa_search_provider(**kwargs: object) -> ExaSearchProviderIntegration:
    """Compatibility shim — constructs ExaSearchProviderIntegration from legacy runtime."""
    runtime = _legacy_create_exa_search_provider(**kwargs)
    if isinstance(runtime, ExaSearchProviderIntegration):
        return runtime
    return ExaSearchProviderIntegration.from_client(runtime)
