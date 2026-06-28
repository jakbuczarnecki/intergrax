# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_milvus_vector_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.milvus.integration import (
    MILVUS_VECTOR_STORE_PROVIDER_ID,
    MilvusVectorStoreIntegration,
    MilvusVectorStoreIntegrationConfig,
    MilvusVectorStoreClient,
)

__all__ = [
    "create_milvus_vector_store",
    "create_milvus_vector_store_integration",
]


def create_milvus_vector_store_integration(
    *,
    client: MilvusVectorStoreClient | None = None,
    enabled: bool = False,
) -> MilvusVectorStoreIntegration:
    """
    Build a contract-based Milvus vector store integration.

    The legacy facade (create_milvus_vector_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Milvus vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MilvusVectorStoreIntegration.from_client(client, enabled=enabled)
    return MilvusVectorStoreIntegration.for_provider(
        provider_id=MILVUS_VECTOR_STORE_PROVIDER_ID,
        display_name="Milvus",
        config=MilvusVectorStoreIntegrationConfig(enabled=enabled),
    )
