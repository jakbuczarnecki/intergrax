# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_huggingface_hub_object_storage as _legacy_create_huggingface_hub_object_storage

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.object_storage.huggingface_hub.integration import (
    HUGGINGFACE_HUB_OBJECT_STORAGE_PROVIDER_ID,
    HuggingfaceHubObjectStorageIntegration,
    HuggingfaceHubObjectStorageIntegrationConfig,
    HuggingfaceHubObjectStorageClient,
)

__all__ = [
    "create_huggingface_hub_object_storage",
    "create_huggingface_hub_object_storage_integration",
]


def create_huggingface_hub_object_storage_integration(
    *,
    client: HuggingfaceHubObjectStorageClient | None = None,
    enabled: bool = False,
) -> HuggingfaceHubObjectStorageIntegration:
    """
    Build a contract-based Huggingface Hub object storage integration.

    The legacy facade (create_huggingface_hub_object_storage) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Huggingface Hub object storage integration requires an injected client when enabled=True",
        )
    if client is not None:
        return HuggingfaceHubObjectStorageIntegration.from_client(client, enabled=enabled)
    return HuggingfaceHubObjectStorageIntegration.for_provider(
        provider_id=HUGGINGFACE_HUB_OBJECT_STORAGE_PROVIDER_ID,
        display_name="Huggingface Hub",
        config=HuggingfaceHubObjectStorageIntegrationConfig(enabled=enabled),
    )


def create_huggingface_hub_object_storage(**kwargs: object) -> HuggingfaceHubObjectStorageIntegration:
    """Compatibility shim — constructs HuggingfaceHubObjectStorageIntegration from legacy runtime."""
    runtime = _legacy_create_huggingface_hub_object_storage(**kwargs)
    if isinstance(runtime, HuggingfaceHubObjectStorageIntegration):
        return runtime
    return HuggingfaceHubObjectStorageIntegration.from_client(runtime)
