# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_supabase_relational_store as _legacy_create_supabase_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.supabase.integration import (
    SUPABASE_RELATIONAL_STORE_PROVIDER_ID,
    SupabaseRelationalStoreIntegration,
    SupabaseRelationalStoreIntegrationConfig,
    SupabaseRelationalStoreClient,
)

__all__ = [
    "create_supabase_relational_store",
    "create_supabase_relational_store_integration",
]


def create_supabase_relational_store_integration(
    *,
    client: SupabaseRelationalStoreClient | None = None,
    enabled: bool = False,
) -> SupabaseRelationalStoreIntegration:
    """
    Build a contract-based Supabase relational store integration.

    The legacy facade (create_supabase_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Supabase relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SupabaseRelationalStoreIntegration.from_client(client, enabled=enabled)
    return SupabaseRelationalStoreIntegration.for_provider(
        provider_id=SUPABASE_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Supabase",
        config=SupabaseRelationalStoreIntegrationConfig(enabled=enabled),
    )


def create_supabase_relational_store(**kwargs: object) -> SupabaseRelationalStoreIntegration:
    """Compatibility shim — constructs SupabaseRelationalStoreIntegration from legacy runtime."""
    runtime = _legacy_create_supabase_relational_store(**kwargs)
    if isinstance(runtime, SupabaseRelationalStoreIntegration):
        return runtime
    return SupabaseRelationalStoreIntegration.from_runtime(runtime)
