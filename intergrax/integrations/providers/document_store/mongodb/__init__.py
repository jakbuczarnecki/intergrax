# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB document store integration (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.document_store.mongodb.config import (
    ENV_MONGODB_COLLECTION,
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
    MongoDBIntegrationConfig,
)

__all__ = [
    "ENV_MONGODB_COLLECTION",
    "ENV_MONGODB_DATABASE",
    "ENV_MONGODB_URI",
    "MongoDBDocumentStore",
    "MongoDBIntegrationBundle",
    "MongoDBIntegrationConfig",
    "create_mongodb_document_store",
    "create_mongodb_integration",
    "register_mongodb_integration",
    "resolve_mongodb_config",
    "create_mongodb_document_store_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "MongoDBIntegrationBundle",
        "MongoDBDocumentStore",
        "create_mongodb_integration",
        "create_mongodb_document_store",
        "register_mongodb_integration",
        "resolve_mongodb_config",
        "create_mongodb_document_store_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MONGODB_DOCUMENT_STORE_PROVIDER_ID",
        "MongodbDocumentStoreIntegration",
        "MongodbDocumentStoreIntegrationConfig",
        "MongodbDocumentStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_mongodb_integration":
        from intergrax.integrations.providers.document_store.mongodb.register import register_mongodb_integration

        return register_mongodb_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.document_store.mongodb import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "MongoDBDocumentStore":
        from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore

        return MongoDBDocumentStore
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.document_store.mongodb import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
