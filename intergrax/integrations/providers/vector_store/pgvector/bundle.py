# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_pgvector_vector_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.pgvector.integration import (
    PGVECTOR_VECTOR_STORE_PROVIDER_ID,
    PgvectorVectorStoreIntegration,
    PgvectorVectorStoreIntegrationConfig,
    PgvectorVectorStoreClient,
)

__all__ = [
    "create_pgvector_vector_store",
    "create_pgvector_vector_store_integration",
]


def create_pgvector_vector_store_integration(
    *,
    client: PgvectorVectorStoreClient | None = None,
    enabled: bool = False,
) -> PgvectorVectorStoreIntegration:
    """
    Build a contract-based pgvector vector store integration.

    The legacy facade (create_pgvector_vector_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "pgvector vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PgvectorVectorStoreIntegration.from_client(client, enabled=enabled)
    return PgvectorVectorStoreIntegration.for_provider(
        provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
        display_name="pgvector",
        config=PgvectorVectorStoreIntegrationConfig(enabled=enabled),
    )
