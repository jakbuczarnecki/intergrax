# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "INMEMORY_VECTOR_STORE_PROVIDER_ID",
    "InmemoryVectorStoreIntegration",
    "InmemoryVectorStoreIntegrationConfig",
    "InmemoryVectorStoreClient",
    "create_inmemory_vector_store",
    "create_inmemory_vector_store_integration",
    "register_inmemory_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_inmemory_vector_store",
        "create_inmemory_vector_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "INMEMORY_VECTOR_STORE_PROVIDER_ID",
        "InmemoryVectorStoreIntegration",
        "InmemoryVectorStoreIntegrationConfig",
        "InmemoryVectorStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "INMEMORY_VECTOR_STORE_PROVIDER_ID",
        "InmemoryVectorStoreIntegration",
        "InmemoryVectorStoreIntegrationConfig",
        "InmemoryVectorStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_inmemory_integration":
        from intergrax.integrations.providers.vector_store.inmemory.register import register_inmemory_integration

        return register_inmemory_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.vector_store.inmemory import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.inmemory import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.vector_store.inmemory import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
