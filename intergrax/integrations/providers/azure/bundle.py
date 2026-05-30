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
from intergrax.integrations.providers.azure.adapter import AzureCloudPlatform
from intergrax.integrations.providers.azure.config import AzureIntegrationConfig
from intergrax.integrations.providers.azure.opens import open_azure_cloud_platform


@dataclass(frozen=True)
class AzureIntegrationBundle:
    config: AzureIntegrationConfig
    cloud_platform: AzureCloudPlatform


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
    assert isinstance(platform, AzureCloudPlatform)
    return AzureIntegrationBundle(config=config, cloud_platform=platform)


def create_azure_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    credential: Optional[object] = None,
    credential_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> AzureCloudPlatform:
    """Catalog factory for ``IntegrationSlug.AZURE`` / ``CLOUD_PLATFORM``."""
    return create_azure_integration(
        cloud_platform=cloud_platform,
        credential=credential,
        credential_factory=credential_factory,
        **config_overrides,
    ).cloud_platform
