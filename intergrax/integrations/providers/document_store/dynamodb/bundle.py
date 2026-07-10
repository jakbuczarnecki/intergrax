# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_dynamodb_document_store as _legacy_create_dynamodb_document_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.dynamodb.integration import (
    DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
    DynamoDBDocumentStoreIntegration,
    DynamoDBDocumentStoreIntegrationConfig,
    DynamoDBDocumentStoreClient,
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
    client: DynamoDBDocumentStoreClient | None = None,
    enabled: bool = False,
) -> DynamoDBDocumentStoreIntegration:
    """
    Build a contract-based DynamoDB document store integration.

    The legacy facade (create_dynamodb_document_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "DynamoDB document store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DynamoDBDocumentStoreIntegration.from_client(client, enabled=enabled)
    return DynamoDBDocumentStoreIntegration.for_provider(
        provider_id=DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
        display_name="DynamoDB",
        config=DynamoDBDocumentStoreIntegrationConfig(enabled=enabled),
    )


def create_dynamodb_document_store(**kwargs: object) -> DocumentStore:
    """Compatibility shim — constructs DynamoDB DocumentStore from legacy runtime."""
    runtime = _legacy_create_dynamodb_document_store(**kwargs)
    if isinstance(runtime, DynamoDBDocumentStoreIntegration):
        return runtime.as_document_store()
    if isinstance(runtime, DynamodbDocumentStoreIntegration):
        return runtime.as_document_store()
    return DynamoDBDocumentStoreIntegration.from_client(runtime).as_document_store()
