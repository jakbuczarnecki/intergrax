# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dynamodb document store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import DocumentStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DYNAMODB_DOCUMENT_STORE_PROVIDER_ID = "dynamodb"


class DynamodbDocumentStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Dynamodb document store integration."""

    pass


@runtime_checkable
class DynamodbDocumentStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DynamodbDocumentStoreIntegration(DocumentStoreIntegrationContract):
    """
    Dynamodb document store integration.

    The legacy facade (create_dynamodb_document_store) remains separate and backward-compatible.
    """

    config: DynamodbDocumentStoreIntegrationConfig = DynamodbDocumentStoreIntegrationConfig()
    _client: DynamodbDocumentStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: DynamodbDocumentStoreClient,
        *,
        enabled: bool = False,
    ) -> DynamodbDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
            display_name="Dynamodb",
            config=DynamodbDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DynamodbDocumentStoreClient | None:
        return self._client
