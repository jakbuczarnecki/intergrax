# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete GCP integration bundle — the single composition root for GCP in Intergrax.

Google credentials are opened only in ``opens.py``. Tier-3 code MUST use
``create_gcp_cloud_platform()``, ``create_gcp_integration()``, or
``profile.resolve(IntegrationCategory.CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.cloud_platform.gcp.config import GcpIntegrationConfig
from intergrax.integrations.providers.cloud_platform.gcp.integration import (
    GCP_CLOUD_PLATFORM_PROVIDER_ID,
    GcpCloudPlatformClient,
    GcpCloudPlatformIntegration,
    GcpCloudPlatformIntegrationConfig,
)
from intergrax.integrations.providers.cloud_platform.gcp.opens import open_gcp_cloud_platform


@dataclass(frozen=True)
class GcpIntegrationBundle:
    config: GcpIntegrationConfig
    cloud_platform: GcpCloudPlatformIntegration


def resolve_gcp_config(**overrides: object) -> GcpIntegrationConfig:
    return GcpIntegrationConfig.from_env(**overrides)


def create_gcp_integration(
    *,
    cloud_platform: CloudPlatform | None = None,
    credentials: object | None = None,
    resolved_project_id: str = "",
    credential_factory: Callable[[], tuple[object, str]] | None = None,
    **config_overrides: object,
) -> GcpIntegrationBundle:
    config = resolve_gcp_config(**config_overrides)
    platform = open_gcp_cloud_platform(
        config,
        implementation=cloud_platform,
        credentials=credentials,
        resolved_project_id=resolved_project_id,
        credential_factory=credential_factory,
    )
    assert isinstance(platform, GcpCloudPlatformIntegration)
    return GcpIntegrationBundle(config=config, cloud_platform=platform)


def create_gcp_cloud_platform_integration(
    *,
    client: GcpCloudPlatformClient | None = None,
    enabled: bool = False,
) -> GcpCloudPlatformIntegration:
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "GCP cloud platform integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GcpCloudPlatformIntegration.from_client(client, enabled=enabled)
    return GcpCloudPlatformIntegration.for_provider(
        provider_id=GCP_CLOUD_PLATFORM_PROVIDER_ID,
        display_name="GCP",
        config=GcpCloudPlatformIntegrationConfig(enabled=enabled),
    )


def create_gcp_cloud_platform(
    *,
    cloud_platform: CloudPlatform | None = None,
    credentials: object | None = None,
    resolved_project_id: str = "",
    credential_factory: Callable[[], tuple[object, str]] | None = None,
    **config_overrides: object,
) -> GcpCloudPlatformIntegration:
    """Catalog factory for ``"gcp"`` / ``CLOUD_PLATFORM``."""
    if (
        cloud_platform is None
        and credentials is None
        and credential_factory is None
        and resolved_project_id == ""
        and not config_overrides
    ):
        return create_gcp_cloud_platform_integration(enabled=False)
    return create_gcp_integration(
        cloud_platform=cloud_platform,
        credentials=credentials,
        resolved_project_id=resolved_project_id,
        credential_factory=credential_factory,
        **config_overrides,
    ).cloud_platform
