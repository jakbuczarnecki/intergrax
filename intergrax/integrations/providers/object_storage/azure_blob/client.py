# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Blob duck-typed client wrapper (no azure SDK here)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Mapping, Optional

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage, is_blob_not_found
from intergrax.integrations.providers.object_storage.azure_blob.config import AzureBlobIntegrationConfig


class AzureBlobClient:
    def __init__(self, config: AzureBlobIntegrationConfig, container_client: Any) -> None:
        self._config = config
        self._container = container_client

    def upload_blob(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        kwargs: dict[str, Any] = {"overwrite": True}
        content_settings = attribute_access.optional(self._container, "content_settings", None)
        if content_settings is not None:
            kwargs["content_settings"] = content_settings(content_type=content_type)
        elif metadata:
            kwargs["metadata"] = dict(metadata)
        self._container.upload_blob(name=key, data=body, **kwargs)

    def download_blob(self, key: str) -> tuple[bytes, str, dict[str, str]]:
        blob = self._container.download_blob(key)
        raw = blob.readall() if hasattr(blob, "readall") else blob.read()
        props = attribute_access.optional(blob, "properties", None) or {}
        content_type = str(attribute_access.optional(props, "content_settings", None) and props.content_settings.content_type or "application/octet-stream")
        metadata = dict(attribute_access.optional(props, "metadata", None) or {})
        return raw, content_type, metadata

    def delete_blob(self, key: str) -> None:
        self._container.delete_blob(key)

    def generate_sas_url(self, key: str, *, expires_in_seconds: int, method: str) -> str:
        generate = attribute_access.optional(self._container, "generate_sas_url", None)
        if callable(generate):
            return str(generate(key, expires_in_seconds=expires_in_seconds, method=method))
        raise NotImplementedError("container client does not support SAS URLs")


def build_azure_blob_object_storage(
    config: AzureBlobIntegrationConfig,
    container_client: Any,
) -> CatalogObjectStorage:
    return CatalogObjectStorage(
        config,
        AzureBlobClient(config, container_client),
        factory_name="create_azure_blob_object_storage",
    )
