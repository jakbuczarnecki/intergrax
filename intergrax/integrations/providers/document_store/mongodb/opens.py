# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level MongoDB connection openers — internal to the mongodb integration package.

Only this module may construct ``pymongo.MongoClient``. All composition roots use
``bundle.create_mongodb_*`` or ``profile.resolve(DOCUMENT_STORE)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from intergrax.integrations.providers.document_store.mongodb.integration import MongoDBDocumentStoreIntegration
from intergrax.integrations.providers.document_store.mongodb.client import MongoCollectionClient
from intergrax.integrations.providers.document_store.mongodb.config import MongoDBIntegrationConfig

DOCUMENT_KEY_INDEX_NAME = "uq_intergrax_document_key"
DOCUMENT_KEY_INDEX_KEYS: tuple[tuple[str, int], ...] = (
    ("partition_key", 1),
    ("row_key", 1),
)


def _import_pymongo() -> Any:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "MongoDB integration requires pymongo. "
            "Install with: uv sync --extra dev  (includes pymongo)"
        ) from exc
    return MongoClient


def _is_duplicate_key_error(exc: BaseException) -> bool:
    try:
        from pymongo.errors import DuplicateKeyError
    except ImportError:
        return False
    return isinstance(exc, DuplicateKeyError)


def _ensure_document_key_index(collection: Any) -> None:
    try:
        collection.create_index(
            list(DOCUMENT_KEY_INDEX_KEYS),
            unique=True,
            name=DOCUMENT_KEY_INDEX_NAME,
        )
    except Exception:
        raise IntegrationConfigurationError(
            "MongoDB document store requires unique compound index "
            f"{DOCUMENT_KEY_INDEX_NAME} on (partition_key, row_key); "
            "index creation failed"
        ) from None


def _open_collection(
    config: MongoDBIntegrationConfig,
    *,
    collection_factory: Optional[Callable[[], Any]] = None,
) -> tuple[Any, Any | None]:
    if collection_factory is not None:
        return collection_factory(), None
    MongoClient = _import_pymongo()
    database_name, collection_name = config.qualified_collection()
    client = MongoClient(config.uri)
    collection = client[database_name][collection_name]
    return collection, client


def open_mongodb_collection_client(
    config: MongoDBIntegrationConfig,
    *,
    collection: Optional[Any] = None,
    client: Optional[Any] = None,
    collection_factory: Optional[Callable[[], Any]] = None,
) -> MongoCollectionClient:
    if collection is not None:
        opened_collection = collection
        opened_client = client
    elif collection_factory is not None:
        opened_collection = collection_factory()
        opened_client = client
    else:
        opened_collection, opened_client = _open_collection(config)
    _ensure_document_key_index(opened_collection)
    return MongoCollectionClient(
        config,
        collection=opened_collection,
        client=opened_client,
        is_duplicate_key_error=_is_duplicate_key_error,
    )


def open_mongodb_document_store(
    config: MongoDBIntegrationConfig,
    *,
    implementation: Optional[DocumentStore] = None,
    collection: Optional[Any] = None,
    client: Optional[Any] = None,
    collection_factory: Optional[Callable[[], Any]] = None,
) -> MongoDBDocumentStoreIntegration:
    if implementation is not None:
        if isinstance(implementation, MongoDBDocumentStoreIntegration):
            return implementation
        return MongoDBDocumentStoreIntegration.from_client(implementation)
    mongo_client = open_mongodb_collection_client(
        config,
        collection=collection,
        client=client,
        collection_factory=collection_factory,
    )
    return MongoDBDocumentStoreIntegration.from_client(_MongoDBDocumentStore(mongo_client))
