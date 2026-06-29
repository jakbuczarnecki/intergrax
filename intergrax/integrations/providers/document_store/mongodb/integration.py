# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mongodb document store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
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
    Single public Mongodb document store entrypoint.

    Legacy catalog factory (create_mongodb_integration) delegates to this class.
    """

    config: MongodbDocumentStoreIntegrationConfig = MongodbDocumentStoreIntegrationConfig()
    _client: MongodbDocumentStoreClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> MongodbDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
            display_name="Mongodb",
            config=MongodbDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Mongodb integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

DocumentStore.register(MongodbDocumentStoreIntegration)
