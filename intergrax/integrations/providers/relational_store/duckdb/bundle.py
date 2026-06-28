# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_duckdb_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.duckdb.integration import (
    DUCKDB_RELATIONAL_STORE_PROVIDER_ID,
    DuckdbRelationalStoreIntegration,
    DuckdbRelationalStoreIntegrationConfig,
    DuckdbRelationalStoreClient,
)

__all__ = [
    "create_duckdb_relational_store",
    "create_duckdb_relational_store_integration",
]


def create_duckdb_relational_store_integration(
    *,
    client: DuckdbRelationalStoreClient | None = None,
    enabled: bool = False,
) -> DuckdbRelationalStoreIntegration:
    """
    Build a contract-based Duckdb relational store integration.

    The legacy facade (create_duckdb_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Duckdb relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DuckdbRelationalStoreIntegration.from_client(client, enabled=enabled)
    return DuckdbRelationalStoreIntegration.for_provider(
        provider_id=DUCKDB_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Duckdb",
        config=DuckdbRelationalStoreIntegrationConfig(enabled=enabled),
    )
