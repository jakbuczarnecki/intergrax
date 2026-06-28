# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_snowflake_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.snowflake.integration import (
    SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID,
    SnowflakeRelationalStoreIntegration,
    SnowflakeRelationalStoreIntegrationConfig,
    SnowflakeRelationalStoreClient,
)

__all__ = [
    "create_snowflake_relational_store",
    "create_snowflake_relational_store_integration",
]


def create_snowflake_relational_store_integration(
    *,
    client: SnowflakeRelationalStoreClient | None = None,
    enabled: bool = False,
) -> SnowflakeRelationalStoreIntegration:
    """
    Build a contract-based Snowflake relational store integration.

    The legacy facade (create_snowflake_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Snowflake relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SnowflakeRelationalStoreIntegration.from_client(client, enabled=enabled)
    return SnowflakeRelationalStoreIntegration.for_provider(
        provider_id=SNOWFLAKE_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Snowflake",
        config=SnowflakeRelationalStoreIntegrationConfig(enabled=enabled),
    )
