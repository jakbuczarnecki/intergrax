# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backblaze B2 object storage integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.object_storage import ObjectStorage, PresignedUrlMethod, StoredObject
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID = "backblaze_b2"


class BackblazeB2ObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Backblaze B2 object storage integration."""

    pass


@runtime_checkable
class BackblazeB2ObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BackblazeB2ObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Single public Backblaze B2 object storage entrypoint.

    Legacy catalog factory (create_backblaze_b2_object_storage) delegates to this class.
    """

    config: BackblazeB2ObjectStorageIntegrationConfig = BackblazeB2ObjectStorageIntegrationConfig()
    _client: BackblazeB2ObjectStorageClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> BackblazeB2ObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Backblaze B2",
            config=BackblazeB2ObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._require_runtime().put(key, body, content_type=content_type, metadata=metadata)

    def get(self, key: str) -> StoredObject | None:
        return self._require_runtime().get(key)

    def delete(self, key: str) -> None:
        self._require_runtime().delete(key)

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: PresignedUrlMethod = "GET",
    ) -> str:
        return self._require_runtime().presigned_url(
            key,
            expires_in_seconds=expires_in_seconds,
            method=method,
        )

    def close(self) -> None:
        self._require_runtime().close()


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


    @classmethod
    def from_client(
        cls,
        client: BackblazeB2ObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> BackblazeB2ObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Backblaze B2",
            config=BackblazeB2ObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BackblazeB2ObjectStorageClient | None:
        return self._client

ObjectStorage.register(BackblazeB2ObjectStorageIntegration)
