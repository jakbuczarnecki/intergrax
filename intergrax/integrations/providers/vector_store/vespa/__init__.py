# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "VESPA_VECTOR_STORE_PROVIDER_ID",
    "VespaVectorStoreIntegration",
    "VespaVectorStoreIntegrationConfig",
    "VespaVectorStoreClient",
    "create_vespa_integration",
    "create_vespa_vector_store",
    "create_vespa_vector_store_integration",
    "register_vespa_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_vespa_integration",
        "create_vespa_vector_store",
        "create_vespa_vector_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "VESPA_VECTOR_STORE_PROVIDER_ID",
        "VespaVectorStoreIntegration",
        "VespaVectorStoreIntegrationConfig",
        "VespaVectorStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "VESPA_VECTOR_STORE_PROVIDER_ID",
        "VespaVectorStoreIntegration",
        "VespaVectorStoreIntegrationConfig",
        "VespaVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_vespa_integration":
        from intergrax.integrations.providers.vector_store.vespa.register import register_vespa_integration

        return register_vespa_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vector_store.vespa import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.vespa import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.vespa import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
