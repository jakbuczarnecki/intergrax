# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID",
    "MotherduckRelationalStoreIntegration",
    "MotherduckRelationalStoreIntegrationConfig",
    "MotherduckRelationalStoreClient",
    "create_motherduck_relational_store",
    "create_motherduck_relational_store_integration",
    "register_motherduck_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_motherduck_relational_store",
        "create_motherduck_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID",
        "MotherduckRelationalStoreIntegration",
        "MotherduckRelationalStoreIntegrationConfig",
        "MotherduckRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID",
        "MotherduckRelationalStoreIntegration",
        "MotherduckRelationalStoreIntegrationConfig",
        "MotherduckRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_motherduck_integration":
        from intergrax.integrations.providers.relational_store.motherduck.register import register_motherduck_integration

        return register_motherduck_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.motherduck import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.motherduck import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.motherduck import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
