# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SUPABASE_RELATIONAL_STORE_PROVIDER_ID",
    "SupabaseRelationalStoreIntegration",
    "SupabaseRelationalStoreIntegrationConfig",
    "SupabaseRelationalStoreClient",
    "create_supabase_relational_store",
    "create_supabase_relational_store_integration",
    "register_supabase_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_supabase_relational_store",
        "create_supabase_relational_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SUPABASE_RELATIONAL_STORE_PROVIDER_ID",
        "SupabaseRelationalStoreIntegration",
        "SupabaseRelationalStoreIntegrationConfig",
        "SupabaseRelationalStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SUPABASE_RELATIONAL_STORE_PROVIDER_ID",
        "SupabaseRelationalStoreIntegration",
        "SupabaseRelationalStoreIntegrationConfig",
        "SupabaseRelationalStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_supabase_integration":
        from intergrax.integrations.providers.relational_store.supabase.register import register_supabase_integration

        return register_supabase_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.relational_store.supabase import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.supabase import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.relational_store.supabase import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
