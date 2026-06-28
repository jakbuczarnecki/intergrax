# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID",
    "SemanticScholarSearchProviderIntegration",
    "SemanticScholarSearchProviderIntegrationConfig",
    "SemanticScholarSearchProviderClient",
    "create_semantic_scholar_search_provider",
    "create_semantic_scholar_search_provider_integration",
    "register_semantic_scholar_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_semantic_scholar_search_provider",
        "create_semantic_scholar_search_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID",
        "SemanticScholarSearchProviderIntegration",
        "SemanticScholarSearchProviderIntegrationConfig",
        "SemanticScholarSearchProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID",
        "SemanticScholarSearchProviderIntegration",
        "SemanticScholarSearchProviderIntegrationConfig",
        "SemanticScholarSearchProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_semantic_scholar_integration":
        from intergrax.integrations.providers.search_provider.semantic_scholar.register import register_semantic_scholar_integration

        return register_semantic_scholar_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.search_provider.semantic_scholar import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.semantic_scholar import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.search_provider.semantic_scholar import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
