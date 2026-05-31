# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared ``ObjectStorage`` facade for blob backends (S3-compatible duck types)."""

from __future__ import annotations

from typing import Mapping, Optional, Protocol

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.object_storage import PresignedUrlMethod, StoredObject


class ObjectKeyConfig(Protocol):
    def object_key(self, key: str) -> str: ...


def is_blob_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        return code in {"NoSuchKey", "404", "NotFound", "BlobNotFound", "ResourceNotFoundException"}
    status = getattr(exc, "status_code", None)
    return status == 404


class CatalogObjectStorage:
    """Catalog ``ObjectStorage`` over a duck-typed blob client."""

    def __init__(
        self,
        config: ObjectKeyConfig,
        client: object,
        *,
        container_attr: str = "container",
        factory_name: str = "create_object_storage",
    ) -> None:
        self._config = config
        self._client = client
        self._closed = False
        self._factory_name = factory_name

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._require_open()
        object_key = self._config.object_key(key)
        upload = getattr(self._client, "upload_blob", None)
        if callable(upload):
            upload(object_key, body, content_type=content_type, metadata=dict(metadata or {}))
            return
        put_object = getattr(self._client, "put_object", None)
        if callable(put_object):
            kwargs = {
                "Key": object_key,
                "Body": body,
                "ContentType": content_type,
            }
            if metadata:
                kwargs["Metadata"] = dict(metadata)
            put_object(**kwargs)
            return
        raise IntegrationConfigurationError(f"{self._factory_name}: blob client missing upload API")

    def get(self, key: str) -> Optional[StoredObject]:
        self._require_open()
        object_key = self._config.object_key(key)
        try:
            download = getattr(self._client, "download_blob", None)
            if callable(download):
                raw, content_type, user_metadata = download(object_key)
                return StoredObject(
                    key=key,
                    body=raw,
                    content_type=content_type or "application/octet-stream",
                    metadata=dict(user_metadata or {}),
                    size_bytes=len(raw),
                )
            get_object = getattr(self._client, "get_object", None)
            if callable(get_object):
                response = get_object(Key=object_key)
                body_stream = response.get("Body")
                raw = body_stream.read() if body_stream is not None else b""
                return StoredObject(
                    key=key,
                    body=raw,
                    content_type=str(response.get("ContentType") or "application/octet-stream"),
                    metadata=dict(response.get("Metadata") or {}),
                    size_bytes=len(raw),
                )
        except Exception as exc:  # noqa: BLE001
            if is_blob_not_found(exc):
                return None
            raise
        return None

    def delete(self, key: str) -> None:
        self._require_open()
        object_key = self._config.object_key(key)
        delete_blob = getattr(self._client, "delete_blob", None)
        if callable(delete_blob):
            delete_blob(object_key)
            return
        delete_object = getattr(self._client, "delete_object", None)
        if callable(delete_object):
            delete_object(Key=object_key)
            return
        raise IntegrationConfigurationError(f"{self._factory_name}: blob client missing delete API")

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: PresignedUrlMethod = "GET",
    ) -> str:
        self._require_open()
        object_key = self._config.object_key(key)
        presign = getattr(self._client, "generate_presigned_url", None)
        if callable(presign):
            client_method = "get_object" if method == "GET" else "put_object"
            return str(
                presign(
                    ClientMethod=client_method,
                    Params={"Key": object_key},
                    ExpiresIn=expires_in_seconds,
                )
            )
        sas = getattr(self._client, "generate_sas_url", None)
        if callable(sas):
            return str(sas(object_key, expires_in_seconds=expires_in_seconds, method=method))
        signed = getattr(self._client, "generate_signed_url", None)
        if callable(signed):
            return str(signed(object_key, expiration=expires_in_seconds, method=method))
        raise IntegrationConfigurationError(f"{self._factory_name}: blob client missing presign API")

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError(
                f"Object storage is closed; create a new store via {self._factory_name}()"
            )
