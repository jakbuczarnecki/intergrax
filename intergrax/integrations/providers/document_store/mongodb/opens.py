# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level MongoDB connection openers — internal to the mongodb integration package.

Only this module may construct ``pymongo.MongoClient``. All composition roots use
``bundle.create_mongodb_*`` or ``profile.resolve(DOCUMENT_STORE)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.adapter import MongoDBDocumentStore
from intergrax.integrations.providers.document_store.mongodb.client import MongoCollectionClient
from intergrax.integrations.providers.document_store.mongodb.config import MongoDBIntegrationConfig


def _import_pymongo() -> Any:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "MongoDB integration requires pymongo. "
            "Install with: uv sync --extra dev  (includes pymongo)"
        ) from exc
    return MongoClient


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
        return MongoCollectionClient(config, collection=collection, client=client)
    if collection_factory is not None:
        return MongoCollectionClient(config, collection=collection_factory(), client=client)
    opened_collection, opened_client = _open_collection(config)
    return MongoCollectionClient(config, collection=opened_collection, client=opened_client)


def open_mongodb_document_store(
    config: MongoDBIntegrationConfig,
    *,
    implementation: Optional[DocumentStore] = None,
    collection: Optional[Any] = None,
    client: Optional[Any] = None,
    collection_factory: Optional[Callable[[], Any]] = None,
) -> DocumentStore:
    if implementation is not None:
        return implementation
    mongo_client = open_mongodb_collection_client(
        config,
        collection=collection,
        client=client,
        collection_factory=collection_factory,
    )
    return MongoDBDocumentStore(mongo_client)
