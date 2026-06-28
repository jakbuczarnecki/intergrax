# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_gcs_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.gcs.integration import (
    GCS_OBJECT_STORAGE_PROVIDER_ID,
    GcsObjectStorageIntegration,
    GcsObjectStorageIntegrationConfig,
    GcsObjectStorageClient,
)

__all__ = [
    "create_gcs_object_storage",
    "create_gcs_object_storage_integration",
]


def create_gcs_object_storage_integration(
    *,
    client: GcsObjectStorageClient | None = None,
    enabled: bool = False,
) -> GcsObjectStorageIntegration:
    """
    Build a contract-based Gcs object storage integration.

    The legacy facade (create_gcs_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Gcs object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GcsObjectStorageIntegration.from_client(client, enabled=enabled)
    return GcsObjectStorageIntegration.for_provider(
        provider_id=GCS_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Gcs",
        config=GcsObjectStorageIntegrationConfig(enabled=enabled),
    )
