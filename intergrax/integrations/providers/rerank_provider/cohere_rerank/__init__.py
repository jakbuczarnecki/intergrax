# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID",
    "CohereRerankRerankProviderIntegration",
    "CohereRerankRerankProviderIntegrationConfig",
    "CohereRerankRerankProviderClient",
    "create_cohere_rerank_provider",
    "create_cohere_rerank_rerank_provider_integration",
    "register_cohere_rerank_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_cohere_rerank_provider",
        "create_cohere_rerank_rerank_provider_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID",
        "CohereRerankRerankProviderIntegration",
        "CohereRerankRerankProviderIntegrationConfig",
        "CohereRerankRerankProviderClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID",
        "CohereRerankRerankProviderIntegration",
        "CohereRerankRerankProviderIntegrationConfig",
        "CohereRerankRerankProviderClient",
    }
)

def __getattr__(name: str):
    if name == "register_cohere_rerank_integration":
        from intergrax.integrations.providers.rerank_provider.cohere_rerank.register import register_cohere_rerank_integration

        return register_cohere_rerank_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.rerank_provider.cohere_rerank import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.rerank_provider.cohere_rerank import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.rerank_provider.cohere_rerank import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
