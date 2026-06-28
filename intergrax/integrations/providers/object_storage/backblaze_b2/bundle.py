# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_backblaze_b2_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.backblaze_b2.integration import (
    BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
    BackblazeB2ObjectStorageIntegration,
    BackblazeB2ObjectStorageIntegrationConfig,
    BackblazeB2ObjectStorageClient,
)

__all__ = [
    "create_backblaze_b2_object_storage",
    "create_backblaze_b2_object_storage_integration",
]


def create_backblaze_b2_object_storage_integration(
    *,
    client: BackblazeB2ObjectStorageClient | None = None,
    enabled: bool = False,
) -> BackblazeB2ObjectStorageIntegration:
    """
    Build a contract-based Backblaze B2 object storage integration.

    The legacy facade (create_backblaze_b2_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Backblaze B2 object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BackblazeB2ObjectStorageIntegration.from_client(client, enabled=enabled)
    return BackblazeB2ObjectStorageIntegration.for_provider(
        provider_id=BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Backblaze B2",
        config=BackblazeB2ObjectStorageIntegrationConfig(enabled=enabled),
    )
