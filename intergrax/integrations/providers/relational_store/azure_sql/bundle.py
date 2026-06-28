# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_azure_sql_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.azure_sql.integration import (
    AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
    AzureSqlRelationalStoreIntegration,
    AzureSqlRelationalStoreIntegrationConfig,
    AzureSqlRelationalStoreClient,
)

__all__ = [
    "create_azure_sql_relational_store",
    "create_azure_sql_relational_store_integration",
]


def create_azure_sql_relational_store_integration(
    *,
    client: AzureSqlRelationalStoreClient | None = None,
    enabled: bool = False,
) -> AzureSqlRelationalStoreIntegration:
    """
    Build a contract-based Azure Sql relational store integration.

    The legacy facade (create_azure_sql_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Azure Sql relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AzureSqlRelationalStoreIntegration.from_client(client, enabled=enabled)
    return AzureSqlRelationalStoreIntegration.for_provider(
        provider_id=AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Azure Sql",
        config=AzureSqlRelationalStoreIntegrationConfig(enabled=enabled),
    )
