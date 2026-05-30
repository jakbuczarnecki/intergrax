# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""S3 bucket client — duck-typed boto3 S3 API (no boto3 import here)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.object_storage import PresignedUrlMethod, StoredObject
from intergrax.integrations.providers.s3.config import S3IntegrationConfig


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        return code in {"NoSuchKey", "404", "NotFound"}
    return False


class S3BucketClient:
    """Low-level S3 object operations scoped to one bucket."""

    def __init__(self, config: S3IntegrationConfig, s3_client: Any) -> None:
        self._config = config
        self._client = s3_client
        self._bucket = config.require_bucket()

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def s3_client(self) -> Any:
        return self._client

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        object_key = self._config.object_key(key)
        kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": object_key,
            "Body": body,
            "ContentType": content_type,
        }
        if metadata:
            kwargs["Metadata"] = dict(metadata)
        self._client.put_object(**kwargs)

    def get(self, key: str) -> Optional[StoredObject]:
        object_key = self._config.object_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=object_key)
        except Exception as exc:  # noqa: BLE001 — vendor errors are duck-typed
            if _is_not_found(exc):
                return None
            raise
        body_stream = response.get("Body")
        raw = body_stream.read() if body_stream is not None else b""
        user_metadata = response.get("Metadata") or {}
        content_type = str(response.get("ContentType") or "application/octet-stream")
        return StoredObject(
            key=key,
            body=raw,
            content_type=content_type,
            metadata=dict(user_metadata),
            size_bytes=len(raw),
        )

    def delete(self, key: str) -> None:
        object_key = self._config.object_key(key)
        self._client.delete_object(Bucket=self._bucket, Key=object_key)

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: PresignedUrlMethod = "GET",
    ) -> str:
        object_key = self._config.object_key(key)
        client_method = "get_object" if method == "GET" else "put_object"
        return str(
            self._client.generate_presigned_url(
                ClientMethod=client_method,
                Params={"Bucket": self._bucket, "Key": object_key},
                ExpiresIn=expires_in_seconds,
            )
        )
