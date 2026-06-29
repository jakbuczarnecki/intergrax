# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloudflare R2 object storage integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.object_storage import ObjectStorage, PresignedUrlMethod, StoredObject
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID = "cloudflare_r2"


class CloudflareR2ObjectStorageIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cloudflare R2 object storage integration."""

    pass


CloudflareR2ObjectStorageClient = ObjectStorage

class CloudflareR2ObjectStorageIntegration(ObjectStorageIntegrationContract):
    """
    Single public Cloudflare R2 object storage entrypoint.

    Legacy catalog factory (create_cloudflare_r2_object_storage) owns catalog behavior; legacy factories use from_client().
    """

    config: CloudflareR2ObjectStorageIntegrationConfig = CloudflareR2ObjectStorageIntegrationConfig()
    _client: CloudflareR2ObjectStorageClient | None = PrivateAttr(default=None)
    


    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._require_client().put(key, body, content_type=content_type, metadata=metadata)

    def get(self, key: str) -> StoredObject | None:
        return self._require_client().get(key)

    def delete(self, key: str) -> None:
        self._require_client().delete(key)

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: PresignedUrlMethod = "GET",
    ) -> str:
        return self._require_client().presigned_url(
            key,
            expires_in_seconds=expires_in_seconds,
            method=method,
        )

    def close(self) -> None:
        self._require_client().close()


    def _require_client(self) -> ObjectStorage:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: CloudflareR2ObjectStorageClient,
        *,
        enabled: bool = False,
    ) -> CloudflareR2ObjectStorageIntegration:
        integration = cls.for_provider(
            provider_id=CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID,
            display_name="Cloudflare R2",
            config=CloudflareR2ObjectStorageIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CloudflareR2ObjectStorageClient | None:
        return self._client

ObjectStorage.register(CloudflareR2ObjectStorageIntegration)
