# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_dynamodb_document_store as _legacy_create_dynamodb_document_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_store.dynamodb.integration import (
    DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
    DynamodbDocumentStoreIntegration,
    DynamodbDocumentStoreIntegrationConfig,
    DynamodbDocumentStoreClient,
)

__all__ = [
    "create_dynamodb_document_store",
    "create_dynamodb_document_store_integration",
]


def create_dynamodb_document_store_integration(
    *,
    client: DynamodbDocumentStoreClient | None = None,
    enabled: bool = False,
) -> DynamodbDocumentStoreIntegration:
    """
    Build a contract-based Dynamodb document store integration.

    The legacy facade (create_dynamodb_document_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Dynamodb document store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DynamodbDocumentStoreIntegration.from_client(client, enabled=enabled)
    return DynamodbDocumentStoreIntegration.for_provider(
        provider_id=DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
        display_name="Dynamodb",
        config=DynamodbDocumentStoreIntegrationConfig(enabled=enabled),
    )


def create_dynamodb_document_store(**kwargs: object) -> DynamodbDocumentStoreIntegration:
    """Compatibility shim — constructs DynamodbDocumentStoreIntegration from legacy runtime."""
    runtime = _legacy_create_dynamodb_document_store(**kwargs)
    if isinstance(runtime, DynamodbDocumentStoreIntegration):
        return runtime
    return DynamodbDocumentStoreIntegration.from_client(runtime)
