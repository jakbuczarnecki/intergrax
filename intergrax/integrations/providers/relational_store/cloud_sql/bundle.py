# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_cloud_sql_relational_store as _legacy_create_cloud_sql_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.cloud_sql.integration import (
    CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
    CloudSqlRelationalStoreIntegration,
    CloudSqlRelationalStoreIntegrationConfig,
    CloudSqlRelationalStoreClient,
)

__all__ = [
    "create_cloud_sql_relational_store",
    "create_cloud_sql_relational_store_integration",
]


def create_cloud_sql_relational_store_integration(
    *,
    client: CloudSqlRelationalStoreClient | None = None,
    enabled: bool = False,
) -> CloudSqlRelationalStoreIntegration:
    """
    Build a contract-based Cloud Sql relational store integration.

    The legacy facade (create_cloud_sql_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Cloud Sql relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CloudSqlRelationalStoreIntegration.from_client(client, enabled=enabled)
    return CloudSqlRelationalStoreIntegration.for_provider(
        provider_id=CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Cloud Sql",
        config=CloudSqlRelationalStoreIntegrationConfig(enabled=enabled),
    )


def create_cloud_sql_relational_store(**kwargs: object) -> CloudSqlRelationalStoreIntegration:
    """Compatibility shim — constructs CloudSqlRelationalStoreIntegration from legacy runtime."""
    runtime = _legacy_create_cloud_sql_relational_store(**kwargs)
    if isinstance(runtime, CloudSqlRelationalStoreIntegration):
        return runtime
    return CloudSqlRelationalStoreIntegration.from_client(runtime)
