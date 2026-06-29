# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Azure integration bundle — the single composition root for Azure in Intergrax.

Azure credentials are opened only in ``opens.py``. Tier-3 code MUST use
``create_azure_cloud_platform()``, ``create_azure_integration()``, or
``profile.resolve(IntegrationCategory.CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.cloud_platform.azure.adapter import _AzureCloudPlatform
from intergrax.integrations.providers.cloud_platform.azure.config import AzureIntegrationConfig
from intergrax.integrations.providers.cloud_platform.azure.opens import open_azure_cloud_platform


@dataclass(frozen=True)
class AzureIntegrationBundle:
    config: AzureIntegrationConfig
    cloud_platform: AzureCloudPlatformIntegration


def resolve_azure_config(**overrides: object) -> AzureIntegrationConfig:
    return AzureIntegrationConfig.from_env(**overrides)


def create_azure_integration(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    credential: Optional[object] = None,
    credential_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> AzureIntegrationBundle:
    config = resolve_azure_config(**config_overrides)
    platform = open_azure_cloud_platform(
        config,
        implementation=cloud_platform,
        credential=credential,
        credential_factory=credential_factory,
    )
    assert isinstance(platform, AzureCloudPlatformIntegration)
    return AzureIntegrationBundle(config=config, cloud_platform=platform)


def create_azure_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    credential: Optional[object] = None,
    credential_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> AzureCloudPlatformIntegration:
    """Catalog factory for ``"azure"`` / ``CLOUD_PLATFORM``."""
    return create_azure_integration(
        cloud_platform=cloud_platform,
        credential=credential,
        credential_factory=credential_factory,
        **config_overrides,
    ).cloud_platform

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.cloud_platform.azure.integration import (
    AZURE_CLOUD_PLATFORM_PROVIDER_ID,
    AzureCloudPlatformIntegration,
    AzureCloudPlatformIntegrationConfig,
    AzureCloudPlatformClient,
)


def create_azure_cloud_platform_integration(
    *,
    client: AzureCloudPlatformIntegrationClient | None = None,
    enabled: bool = False,
) -> AzureCloudPlatformIntegration:
    """
    Build a contract-based Azure cloud platform integration.

    Compatibility shim — constructs Integration via from_store (create_azure_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Azure cloud platform integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AzureCloudPlatformIntegration.from_client(client, enabled=enabled)
    return AzureCloudPlatformIntegration.for_provider(
        provider_id=AZURE_CLOUD_PLATFORM_PROVIDER_ID,
        display_name="Azure",
        config=AzureCloudPlatformIntegrationConfig(enabled=enabled),
    )
