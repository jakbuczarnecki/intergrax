# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_weaviate_vector_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.weaviate.integration import (
    WEAVIATE_VECTOR_STORE_PROVIDER_ID,
    WeaviateVectorStoreIntegration,
    WeaviateVectorStoreIntegrationConfig,
    WeaviateVectorStoreClient,
)

__all__ = [
    "create_weaviate_vector_store",
    "create_weaviate_vector_store_integration",
]


def create_weaviate_vector_store_integration(
    *,
    client: WeaviateVectorStoreClient | None = None,
    enabled: bool = False,
) -> WeaviateVectorStoreIntegration:
    """
    Build a contract-based Weaviate vector store integration.

    The legacy facade (create_weaviate_vector_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Weaviate vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return WeaviateVectorStoreIntegration.from_client(client, enabled=enabled)
    return WeaviateVectorStoreIntegration.for_provider(
        provider_id=WEAVIATE_VECTOR_STORE_PROVIDER_ID,
        display_name="Weaviate",
        config=WeaviateVectorStoreIntegrationConfig(enabled=enabled),
    )
