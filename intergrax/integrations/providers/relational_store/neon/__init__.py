# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "NEON_RELATIONAL_STORE_PROVIDER_ID",
    "NeonRelationalStoreIntegration",
    "NeonRelationalStoreIntegrationConfig",
    "NeonRelationalStoreClient",
    "create_neon_relational_store",
    "create_neon_relational_store_integration",
    "register_neon_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_neon_relational_store",
        "create_neon_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "NEON_RELATIONAL_STORE_PROVIDER_ID",
        "NeonRelationalStoreIntegration",
        "NeonRelationalStoreIntegrationConfig",
        "NeonRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "NEON_RELATIONAL_STORE_PROVIDER_ID",
        "NeonRelationalStoreIntegration",
        "NeonRelationalStoreIntegrationConfig",
        "NeonRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_neon_integration":
        from intergrax.integrations.providers.relational_store.neon.register import register_neon_integration

        return register_neon_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.neon import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.neon import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.neon import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
