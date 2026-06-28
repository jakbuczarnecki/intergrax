# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_bigquery_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.bigquery.integration import (
    BIGQUERY_RELATIONAL_STORE_PROVIDER_ID,
    BigqueryRelationalStoreIntegration,
    BigqueryRelationalStoreIntegrationConfig,
    BigqueryRelationalStoreClient,
)

__all__ = [
    "create_bigquery_relational_store",
    "create_bigquery_relational_store_integration",
]


def create_bigquery_relational_store_integration(
    *,
    client: BigqueryRelationalStoreClient | None = None,
    enabled: bool = False,
) -> BigqueryRelationalStoreIntegration:
    """
    Build a contract-based Bigquery relational store integration.

    The legacy facade (create_bigquery_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Bigquery relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BigqueryRelationalStoreIntegration.from_client(client, enabled=enabled)
    return BigqueryRelationalStoreIntegration.for_provider(
        provider_id=BIGQUERY_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Bigquery",
        config=BigqueryRelationalStoreIntegrationConfig(enabled=enabled),
    )
