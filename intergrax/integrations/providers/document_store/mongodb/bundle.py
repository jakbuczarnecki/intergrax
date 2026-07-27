# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete MongoDB integration bundle — the single composition root for MongoDB in Intergrax.

Driver connections are opened only in ``opens.py``. Tier-3 code MUST use
``create_mongodb_document_store()``, ``create_mongodb_integration()``, or
``profile.resolve(IntegrationCategory.DOCUMENT_STORE)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from intergrax.integrations.providers.document_store.mongodb.client import MongoCollectionClient
from intergrax.integrations.providers.document_store.mongodb.config import MongoDBIntegrationConfig
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
    MongoDBDocumentStoreIntegrationConfig,
    MongodbDocumentStoreClient,
)
from intergrax.integrations.providers.document_store.mongodb.opens import open_mongodb_document_store


@dataclass(frozen=True)
class MongoDBIntegrationBundle:
    config: MongoDBIntegrationConfig
    document_store: MongoDBDocumentStoreIntegration
    collection_client: MongoCollectionClient


def resolve_mongodb_config(**overrides: object) -> MongoDBIntegrationConfig:
    return MongoDBIntegrationConfig.from_env(**overrides)


def create_mongodb_integration(
    *,
    document_store: Optional[DocumentStore] = None,
    collection: Optional[object] = None,
    client: Optional[object] = None,
    collection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> MongoDBIntegrationBundle:
    config = resolve_mongodb_config(**config_overrides)
    integration = open_mongodb_document_store(
        config,
        implementation=document_store,
        collection=collection,
        client=client,
        collection_factory=collection_factory,
    )
    assert isinstance(integration, MongoDBDocumentStoreIntegration)
    adapter = integration.as_document_store()
    assert isinstance(adapter, _MongoDBDocumentStore)
    return MongoDBIntegrationBundle(
        config=config,
        document_store=integration,
        collection_client=adapter.mongo_client,
    )


def create_mongodb_document_store(
    *,
    document_store: Optional[DocumentStore] = None,
    collection: Optional[object] = None,
    client: Optional[object] = None,
    collection_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> DocumentStore:
    """Catalog factory for ``"mongodb"`` / ``DOCUMENT_STORE``."""
    return create_mongodb_integration(
        document_store=document_store,
        collection=collection,
        client=client,
        collection_factory=collection_factory,
        **config_overrides,
    ).document_store.as_document_store()


def create_mongodb_document_store_integration(
    *,
    client: MongodbDocumentStoreClient | None = None,
    enabled: bool = False,
) -> MongoDBDocumentStoreIntegration:
    """
    Build a contract-based MongoDB document store integration.

    Compatibility shim — constructs Integration via from_store (create_mongodb_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "MongoDB document store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MongoDBDocumentStoreIntegration.from_client(client, enabled=enabled)
    return MongoDBDocumentStoreIntegration.for_provider(
        provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
        display_name="MongoDB",
        config=MongoDBDocumentStoreIntegrationConfig(enabled=enabled),
    )
