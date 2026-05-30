# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB document store integration (Phase M.6 P2)."""

from intergrax.integrations.providers.mongodb.config import (
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
]

_LAZY_EXPORTS = frozenset(
    {
        "MongoDBIntegrationBundle",
        "MongoDBDocumentStore",
        "create_mongodb_integration",
        "create_mongodb_document_store",
        "register_mongodb_integration",
        "resolve_mongodb_config",
    }
)


def __getattr__(name: str):
    if name == "register_mongodb_integration":
        from intergrax.integrations.providers.mongodb.register import register_mongodb_integration

        return register_mongodb_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.mongodb import bundle as _bundle

        return getattr(_bundle, name)
    if name == "MongoDBDocumentStore":
        from intergrax.integrations.providers.mongodb.adapter import MongoDBDocumentStore

        return MongoDBDocumentStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
