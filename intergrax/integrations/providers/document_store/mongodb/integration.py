# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mongodb document store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import DocumentStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MONGODB_DOCUMENT_STORE_PROVIDER_ID = "mongodb"


class MongodbDocumentStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mongodb document store integration."""

    pass


@runtime_checkable
class MongodbDocumentStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MongodbDocumentStoreIntegration(DocumentStoreIntegrationContract):
    """
    Mongodb document store integration.

    The legacy facade (create_mongodb_integration) remains separate and backward-compatible.
    """

    config: MongodbDocumentStoreIntegrationConfig = MongodbDocumentStoreIntegrationConfig()
    _client: MongodbDocumentStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MongodbDocumentStoreClient,
        *,
        enabled: bool = False,
    ) -> MongodbDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
            display_name="Mongodb",
            config=MongodbDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MongodbDocumentStoreClient | None:
        return self._client
