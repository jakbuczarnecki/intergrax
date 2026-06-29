# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcs object storage integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.object_storage import ObjectStorage, PresignedUrlMethod, StoredObject
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCS_OBJECT_STORAGE_PROVIDER_ID = "gcs"


class GcsObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcs object storage integration."""

    pass


@runtime_checkable
class GcsObjectStorageClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GcsObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Single public Gcs object storage entrypoint.

    Legacy catalog factory (create_gcs_object_storage) delegates to this class.
    """

    config: GcsObjectStorageIntegrationConfig = GcsObjectStorageIntegrationConfig()
    _client: GcsObjectStorageClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> GcsObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=GCS_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Gcs",
            config=GcsObjectStorageIntegrationConfig(enabled=enabled),
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
        client: GcsObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> GcsObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=GCS_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Gcs",
            config=GcsObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcsObjectStorageClient | None:
        return self._client

ObjectStorage.register(GcsObjectStorageIntegration)
