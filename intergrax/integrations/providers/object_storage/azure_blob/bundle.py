# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, Optional

from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.providers.object_storage.azure_blob.config import AzureBlobIntegrationConfig
from intergrax.integrations.providers.object_storage.azure_blob.opens import open_azure_blob_object_storage


def resolve_azure_blob_config(**overrides: object) -> AzureBlobIntegrationConfig:
    return AzureBlobIntegrationConfig.from_env(**overrides)


def create_azure_blob_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    container_client: Optional[object] = None,
    container_client_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    config = resolve_azure_blob_config(**config_overrides)
    return open_azure_blob_object_storage(
        config,
        implementation=object_storage,
        container_client=container_client,
        container_client_factory=container_client_factory,
    )
