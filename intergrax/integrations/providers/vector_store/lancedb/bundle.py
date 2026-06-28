# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_lancedb_vector_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.lancedb.integration import (
    LANCEDB_VECTOR_STORE_PROVIDER_ID,
    LancedbVectorStoreIntegration,
    LancedbVectorStoreIntegrationConfig,
    LancedbVectorStoreClient,
)

__all__ = [
    "create_lancedb_vector_store",
    "create_lancedb_vector_store_integration",
]


def create_lancedb_vector_store_integration(
    *,
    client: LancedbVectorStoreClient | None = None,
    enabled: bool = False,
) -> LancedbVectorStoreIntegration:
    """
    Build a contract-based Lancedb vector store integration.

    The legacy facade (create_lancedb_vector_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Lancedb vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LancedbVectorStoreIntegration.from_client(client, enabled=enabled)
    return LancedbVectorStoreIntegration.for_provider(
        provider_id=LANCEDB_VECTOR_STORE_PROVIDER_ID,
        display_name="Lancedb",
        config=LancedbVectorStoreIntegrationConfig(enabled=enabled),
    )
