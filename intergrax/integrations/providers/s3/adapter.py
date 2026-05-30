# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""S3 object storage adapter — ``ObjectStorage`` facade (no boto3 I/O here)."""

from __future__ import annotations

from typing import Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.object_storage import PresignedUrlMethod, StoredObject
from intergrax.integrations.providers.s3.client import S3BucketClient


class S3ObjectStorage:
    """
    Catalog facade over ``S3BucketClient``.

    Connections are opened only in ``opens.open_s3_object_storage()``.
    Tier-3 code MUST use ``create_s3_object_storage()`` or ``profile.resolve()``.
    """

    def __init__(self, client: S3BucketClient) -> None:
        self._client = client
        self._closed = False

    @property
    def bucket_client(self) -> S3BucketClient:
        return self._client

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._require_open()
        self._client.put(key, body, content_type=content_type, metadata=metadata)

    def get(self, key: str) -> Optional[StoredObject]:
        self._require_open()
        return self._client.get(key)

    def delete(self, key: str) -> None:
        self._require_open()
        self._client.delete(key)

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: PresignedUrlMethod = "GET",
    ) -> str:
        self._require_open()
        return self._client.presigned_url(
            key,
            expires_in_seconds=expires_in_seconds,
            method=method,
        )

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError(
                "S3 object storage is closed; create a new store via create_s3_object_storage()"
            )
