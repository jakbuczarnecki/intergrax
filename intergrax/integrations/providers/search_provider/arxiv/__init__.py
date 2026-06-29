# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ARXIV_SEARCH_PROVIDER_PROVIDER_ID",
    "ArxivSearchProviderIntegration",
    "ArxivSearchProviderIntegrationConfig",
    "ArxivSearchProviderClient",
    "create_arxiv_search_provider",
    "create_arxiv_search_provider_integration",
    "register_arxiv_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_arxiv_search_provider",
        "create_arxiv_search_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "ARXIV_SEARCH_PROVIDER_PROVIDER_ID",
        "ArxivSearchProviderIntegration",
        "ArxivSearchProviderIntegrationConfig",
        "ArxivSearchProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "ARXIV_SEARCH_PROVIDER_PROVIDER_ID",
        "ArxivSearchProviderIntegration",
        "ArxivSearchProviderIntegrationConfig",
        "ArxivSearchProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_arxiv_integration":
        from intergrax.integrations.providers.search_provider.arxiv.register import register_arxiv_integration

        return register_arxiv_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.search_provider.arxiv import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.arxiv import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.arxiv import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
