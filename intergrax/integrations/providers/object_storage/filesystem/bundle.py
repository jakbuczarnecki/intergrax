# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_filesystem_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.filesystem.integration import (
    FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID,
    FilesystemObjectStorageIntegration,
    FilesystemObjectStorageIntegrationConfig,
    FilesystemObjectStorageClient,
)

__all__ = [
    "create_filesystem_object_storage",
    "create_filesystem_object_storage_integration",
]


def create_filesystem_object_storage_integration(
    *,
    client: FilesystemObjectStorageClient | None = None,
    enabled: bool = False,
) -> FilesystemObjectStorageIntegration:
    """
    Build a contract-based Filesystem object storage integration.

    The legacy facade (create_filesystem_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Filesystem object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return FilesystemObjectStorageIntegration.from_client(client, enabled=enabled)
    return FilesystemObjectStorageIntegration.for_provider(
        provider_id=FILESYSTEM_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Filesystem",
        config=FilesystemObjectStorageIntegrationConfig(enabled=enabled),
    )
