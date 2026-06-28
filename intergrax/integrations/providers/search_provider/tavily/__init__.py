# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "TAVILY_SEARCH_PROVIDER_PROVIDER_ID",
    "TavilySearchProviderIntegration",
    "TavilySearchProviderIntegrationConfig",
    "TavilySearchProviderClient",
    "create_tavily_search_provider",
    "create_tavily_search_provider_integration",
    "register_tavily_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_tavily_search_provider",
        "create_tavily_search_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "TAVILY_SEARCH_PROVIDER_PROVIDER_ID",
        "TavilySearchProviderIntegration",
        "TavilySearchProviderIntegrationConfig",
        "TavilySearchProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "TAVILY_SEARCH_PROVIDER_PROVIDER_ID",
        "TavilySearchProviderIntegration",
        "TavilySearchProviderIntegrationConfig",
        "TavilySearchProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_tavily_integration":
        from intergrax.integrations.providers.search_provider.tavily.register import register_tavily_integration

        return register_tavily_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.search_provider.tavily import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.tavily import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.tavily import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
