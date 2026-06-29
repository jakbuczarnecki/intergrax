# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID",
    "PerplexitySearchProviderIntegration",
    "PerplexitySearchProviderIntegrationConfig",
    "PerplexitySearchProviderClient",
    "create_perplexity_search_provider",
    "create_perplexity_search_provider_integration",
    "register_perplexity_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_perplexity_search_provider",
        "create_perplexity_search_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID",
        "PerplexitySearchProviderIntegration",
        "PerplexitySearchProviderIntegrationConfig",
        "PerplexitySearchProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID",
        "PerplexitySearchProviderIntegration",
        "PerplexitySearchProviderIntegrationConfig",
        "PerplexitySearchProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_perplexity_integration":
        from intergrax.integrations.providers.search_provider.perplexity.register import register_perplexity_integration

        return register_perplexity_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.search_provider.perplexity import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.perplexity import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.perplexity import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
