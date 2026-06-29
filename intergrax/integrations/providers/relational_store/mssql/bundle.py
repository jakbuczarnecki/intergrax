# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_mssql_relational_store as _legacy_create_mssql_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.mssql.integration import (
    MSSQL_RELATIONAL_STORE_PROVIDER_ID,
    MssqlRelationalStoreIntegration,
    MssqlRelationalStoreIntegrationConfig,
    MssqlRelationalStoreClient,
)

__all__ = [
    "create_mssql_relational_store",
    "create_mssql_relational_store_integration",
]


def create_mssql_relational_store_integration(
    *,
    client: MssqlRelationalStoreClient | None = None,
    enabled: bool = False,
) -> MssqlRelationalStoreIntegration:
    """
    Build a contract-based Mssql relational store integration.

    The legacy facade (create_mssql_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Mssql relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MssqlRelationalStoreIntegration.from_client(client, enabled=enabled)
    return MssqlRelationalStoreIntegration.for_provider(
        provider_id=MSSQL_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Mssql",
        config=MssqlRelationalStoreIntegrationConfig(enabled=enabled),
    )


def create_mssql_relational_store(**kwargs: object) -> MssqlRelationalStoreIntegration:
    """Compatibility shim — constructs MssqlRelationalStoreIntegration from legacy runtime."""
    runtime = _legacy_create_mssql_relational_store(**kwargs)
    if isinstance(runtime, MssqlRelationalStoreIntegration):
        return runtime
    return MssqlRelationalStoreIntegration.from_client(runtime)
