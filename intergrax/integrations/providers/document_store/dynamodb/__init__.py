# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DYNAMODB_DOCUMENT_STORE_PROVIDER_ID",
    "DynamodbDocumentStoreIntegration",
    "DynamodbDocumentStoreIntegrationConfig",
    "DynamodbDocumentStoreClient",
    "create_dynamodb_document_store",
    "create_dynamodb_document_store_integration",
    "register_dynamodb_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_dynamodb_document_store",
        "create_dynamodb_document_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DYNAMODB_DOCUMENT_STORE_PROVIDER_ID",
        "DynamodbDocumentStoreIntegration",
        "DynamodbDocumentStoreIntegrationConfig",
        "DynamodbDocumentStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "DYNAMODB_DOCUMENT_STORE_PROVIDER_ID",
        "DynamodbDocumentStoreIntegration",
        "DynamodbDocumentStoreIntegrationConfig",
        "DynamodbDocumentStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_dynamodb_integration":
        from intergrax.integrations.providers.document_store.dynamodb.register import register_dynamodb_integration

        return register_dynamodb_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.document_store.dynamodb import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_store.dynamodb import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_store.dynamodb import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
