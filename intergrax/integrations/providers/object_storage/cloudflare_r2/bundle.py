# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_cloudflare_r2_object_storage as _legacy_create_cloudflare_r2_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.cloudflare_r2.integration import (
    CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID,
    CloudflareR2ObjectStorageIntegration,
    CloudflareR2ObjectStorageIntegrationConfig,
    CloudflareR2ObjectStorageClient,
)

__all__ = [
    "create_cloudflare_r2_object_storage",
    "create_cloudflare_r2_object_storage_integration",
]


def create_cloudflare_r2_object_storage_integration(
    *,
    client: CloudflareR2ObjectStorageClient | None = None,
    enabled: bool = False,
) -> CloudflareR2ObjectStorageIntegration:
    """
    Build a contract-based Cloudflare R2 object storage integration.

    The legacy facade (create_cloudflare_r2_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Cloudflare R2 object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CloudflareR2ObjectStorageIntegration.from_client(client, enabled=enabled)
    return CloudflareR2ObjectStorageIntegration.for_provider(
        provider_id=CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Cloudflare R2",
        config=CloudflareR2ObjectStorageIntegrationConfig(enabled=enabled),
    )


def create_cloudflare_r2_object_storage(**kwargs: object) -> CloudflareR2ObjectStorageIntegration:
    """Compatibility shim — constructs CloudflareR2ObjectStorageIntegration from legacy runtime."""
    runtime = _legacy_create_cloudflare_r2_object_storage(**kwargs)
    if isinstance(runtime, CloudflareR2ObjectStorageIntegration):
        return runtime
    return CloudflareR2ObjectStorageIntegration.from_runtime(runtime)
