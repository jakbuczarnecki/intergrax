# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_typesense_vector_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.typesense.integration import (
    TYPESENSE_VECTOR_STORE_PROVIDER_ID,
    TypesenseVectorStoreIntegration,
    TypesenseVectorStoreIntegrationConfig,
    TypesenseVectorStoreClient,
)

__all__ = [
    "create_typesense_vector_store",
    "create_typesense_vector_store_integration",
]


def create_typesense_vector_store_integration(
    *,
    client: TypesenseVectorStoreClient | None = None,
    enabled: bool = False,
) -> TypesenseVectorStoreIntegration:
    """
    Build a contract-based Typesense vector store integration.

    The legacy facade (create_typesense_vector_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Typesense vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TypesenseVectorStoreIntegration.from_client(client, enabled=enabled)
    return TypesenseVectorStoreIntegration.for_provider(
        provider_id=TYPESENSE_VECTOR_STORE_PROVIDER_ID,
        display_name="Typesense",
        config=TypesenseVectorStoreIntegrationConfig(enabled=enabled),
    )
