# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_minio_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.minio.integration import (
    MINIO_OBJECT_STORAGE_PROVIDER_ID,
    MinioObjectStorageIntegration,
    MinioObjectStorageIntegrationConfig,
    MinioObjectStorageClient,
)

__all__ = [
    "create_minio_object_storage",
    "create_minio_object_storage_integration",
]


def create_minio_object_storage_integration(
    *,
    client: MinioObjectStorageClient | None = None,
    enabled: bool = False,
) -> MinioObjectStorageIntegration:
    """
    Build a contract-based Minio object storage integration.

    The legacy facade (create_minio_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Minio object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MinioObjectStorageIntegration.from_client(client, enabled=enabled)
    return MinioObjectStorageIntegration.for_provider(
        provider_id=MINIO_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Minio",
        config=MinioObjectStorageIntegrationConfig(enabled=enabled),
    )
