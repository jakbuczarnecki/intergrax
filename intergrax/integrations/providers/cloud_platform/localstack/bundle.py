# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_localstack_cloud_platform

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.cloud_platform.localstack.integration import (
    LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID,
    LocalstackCloudPlatformIntegration,
    LocalstackCloudPlatformIntegrationConfig,
    LocalstackCloudPlatformClient,
)

__all__ = [
    "create_localstack_cloud_platform",
    "create_localstack_cloud_platform_integration",
]


def create_localstack_cloud_platform_integration(
    *,
    client: LocalstackCloudPlatformClient | None = None,
    enabled: bool = False,
) -> LocalstackCloudPlatformIntegration:
    """
    Build a contract-based Localstack cloud platform integration.

    The legacy facade (create_localstack_cloud_platform) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Localstack cloud platform integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LocalstackCloudPlatformIntegration.from_client(client, enabled=enabled)
    return LocalstackCloudPlatformIntegration.for_provider(
        provider_id=LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID,
        display_name="Localstack",
        config=LocalstackCloudPlatformIntegrationConfig(enabled=enabled),
    )
