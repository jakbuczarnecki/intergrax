# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GCP Cloud Storage client wrapper (no google SDK here)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Mapping, Optional

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage
from intergrax.integrations.contracts.base import IntegrationConfigurationError


class GcsIntegrationConfigProtocol:
    bucket: str
    prefix: str

    def require_bucket(self) -> str: ...
    def object_key(self, key: str) -> str: ...


class GcsBlobClient:
    def __init__(self, config: GcsIntegrationConfigProtocol, bucket: Any) -> None:
        self._config = config
        self._bucket = bucket

    def upload_blob(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(body, content_type=content_type)
        if metadata:
            blob.metadata = dict(metadata)
            blob.patch()

    def download_blob(self, key: str) -> tuple[bytes, str, dict[str, str]]:
        blob = self._bucket.blob(key)
        if not blob.exists():
            raise FileNotFoundError(key)
        raw = blob.download_as_bytes()
        content_type = str(attribute_access.optional(blob, "content_type", None) or "application/octet-stream")
        return raw, content_type, dict(attribute_access.optional(blob, "metadata", None) or {})

    def delete_blob(self, key: str) -> None:
        self._bucket.blob(key).delete()

    def generate_signed_url(self, key: str, *, expiration: int, method: str) -> str:
        blob = self._bucket.blob(key)
        http_method = "GET" if method == "GET" else "PUT"
        return str(blob.generate_signed_url(expiration=expiration, method=http_method))


def build_gcs_object_storage(config: GcsIntegrationConfigProtocol, bucket: Any) -> CatalogObjectStorage:
    return CatalogObjectStorage(
        config,
        GcsBlobClient(config, bucket),
        factory_name="create_gcs_object_storage",
    )


def open_gcs_bucket(config: GcsIntegrationConfigProtocol) -> Any:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "GCS integration requires google-cloud-storage. "
            "Install with: uv pip install google-cloud-storage"
        ) from exc
    client = storage.Client(project=config.project_id or None)  # type: ignore[attr-defined]
    bucket_name = config.require_bucket()  # type: ignore[attr-defined]
    return client.bucket(bucket_name)
