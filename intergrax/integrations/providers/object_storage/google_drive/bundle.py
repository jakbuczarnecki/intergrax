# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_google_drive_object_storage as _legacy_create_google_drive_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.google_drive.integration import (
    GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID,
    GoogleDriveObjectStorageIntegration,
    GoogleDriveObjectStorageIntegrationConfig,
    GoogleDriveObjectStorageClient,
)

__all__ = [
    "create_google_drive_object_storage",
    "create_google_drive_object_storage_integration",
]


def create_google_drive_object_storage_integration(
    *,
    client: GoogleDriveObjectStorageClient | None = None,
    enabled: bool = False,
) -> GoogleDriveObjectStorageIntegration:
    """
    Build a contract-based Google Drive object storage integration.

    The legacy facade (create_google_drive_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Google Drive object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GoogleDriveObjectStorageIntegration.from_client(client, enabled=enabled)
    return GoogleDriveObjectStorageIntegration.for_provider(
        provider_id=GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Google Drive",
        config=GoogleDriveObjectStorageIntegrationConfig(enabled=enabled),
    )


def create_google_drive_object_storage(**kwargs: object) -> GoogleDriveObjectStorageIntegration:
    """Compatibility shim — constructs GoogleDriveObjectStorageIntegration from legacy runtime."""
    runtime = _legacy_create_google_drive_object_storage(**kwargs)
    if isinstance(runtime, GoogleDriveObjectStorageIntegration):
        return runtime
    return GoogleDriveObjectStorageIntegration.from_runtime(runtime)
