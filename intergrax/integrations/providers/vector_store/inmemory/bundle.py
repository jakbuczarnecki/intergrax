# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_inmemory_vector_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    INMEMORY_VECTOR_STORE_PROVIDER_ID,
    InmemoryVectorStoreIntegration,
    InmemoryVectorStoreIntegrationConfig,
    InmemoryVectorStoreClient,
)

__all__ = [
    "create_inmemory_vector_store",
    "create_inmemory_vector_store_integration",
]


def create_inmemory_vector_store_integration(
    *,
    client: InmemoryVectorStoreClient | None = None,
    enabled: bool = False,
) -> InmemoryVectorStoreIntegration:
    """
    Build a contract-based Inmemory vector store integration.

    The legacy facade (create_inmemory_vector_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Inmemory vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return InmemoryVectorStoreIntegration.from_client(client, enabled=enabled)
    return InmemoryVectorStoreIntegration.for_provider(
        provider_id=INMEMORY_VECTOR_STORE_PROVIDER_ID,
        display_name="Inmemory",
        config=InmemoryVectorStoreIntegrationConfig(enabled=enabled),
    )
