# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Bing integration — single public entry for Bing Web Search.

Implementation lives under ``intergrax.websearch.providers.bing_provider``;
compose only through this package.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.search_provider.bing.config import (
    DEFAULT_TIMEOUT_SECONDS,
    ENV_BING_API_KEY,
    LEGACY_ENV_API_KEY,
    BingIntegrationConfig,
)

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ENV_BING_API_KEY",
    "BingIntegrationBundle",
    "BingIntegrationConfig",
    "BingSearchProvider",
    "LEGACY_ENV_API_KEY",
    "create_bing_integration",
    "create_bing_search_provider",
    "register_bing_integration",
    "resolve_bing_config",
    "create_bing_search_provider_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "BingIntegrationBundle",
        "BingSearchProvider",
        "create_bing_integration",
        "create_bing_search_provider",
        "resolve_bing_config",
        "create_bing_search_provider_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "BING_SEARCH_PROVIDER_PROVIDER_ID",
        "BingSearchProviderIntegration",
        "BingSearchProviderIntegrationConfig",
        "BingSearchProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_bing_integration":
        from intergrax.integrations.providers.search_provider.bing.register import register_bing_integration

        return register_bing_integration
    if name == "BingSearchProvider":
        from intergrax.integrations.providers.search_provider.bing.adapter import BingSearchProvider

        return BingSearchProvider
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.search_provider.bing import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.bing import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
