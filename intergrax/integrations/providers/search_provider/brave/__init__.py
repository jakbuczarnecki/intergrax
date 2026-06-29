# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "BRAVE_SEARCH_PROVIDER_PROVIDER_ID",
    "BraveSearchProviderIntegration",
    "BraveSearchProviderIntegrationConfig",
    "BraveSearchProviderClient",
    "create_brave_search_provider",
    "create_brave_search_provider_integration",
    "register_brave_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_brave_search_provider",
        "create_brave_search_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "BRAVE_SEARCH_PROVIDER_PROVIDER_ID",
        "BraveSearchProviderIntegration",
        "BraveSearchProviderIntegrationConfig",
        "BraveSearchProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "BRAVE_SEARCH_PROVIDER_PROVIDER_ID",
        "BraveSearchProviderIntegration",
        "BraveSearchProviderIntegrationConfig",
        "BraveSearchProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_brave_integration":
        from intergrax.integrations.providers.search_provider.brave.register import register_brave_integration

        return register_brave_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.search_provider.brave import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.brave import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.brave import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
