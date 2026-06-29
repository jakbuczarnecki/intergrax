# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete GCP integration bundle — the single composition root for GCP in Intergrax.

Google credentials are opened only in ``opens.py``. Tier-3 code MUST use
``create_gcp_cloud_platform()``, ``create_gcp_integration()``, or
``profile.resolve(IntegrationCategory.CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.cloud_platform.gcp.adapter import _GcpCloudPlatform
from intergrax.integrations.providers.cloud_platform.gcp.config import GcpIntegrationConfig
from intergrax.integrations.providers.cloud_platform.gcp.opens import open_gcp_cloud_platform


@dataclass(frozen=True)
class GcpIntegrationBundle:
    config: GcpIntegrationConfig
    cloud_platform: GcpCloudPlatformIntegration


def resolve_gcp_config(**overrides: object) -> GcpIntegrationConfig:
    return GcpIntegrationConfig.from_env(**overrides)


def create_gcp_integration(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    credentials: Optional[object] = None,
    resolved_project_id: str = "",
    credential_factory: Optional[Callable[[], tuple[object, str]]] = None,
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


def create_gcp_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    credentials: Optional[object] = None,
    resolved_project_id: str = "",
    credential_factory: Optional[Callable[[], tuple[object, str]]] = None,
    **config_overrides: object,
) -> GcpCloudPlatformIntegration:
    """Catalog factory for ``"gcp"`` / ``CLOUD_PLATFORM``."""
    return create_gcp_integration(
        cloud_platform=cloud_platform,
        credentials=credentials,
        resolved_project_id=resolved_project_id,
        credential_factory=credential_factory,
        **config_overrides,
    ).cloud_platform

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.cloud_platform.gcp.integration import (
    GCP_CLOUD_PLATFORM_PROVIDER_ID,
    GcpCloudPlatformIntegration,
    GcpCloudPlatformIntegrationConfig,
    GcpCloudPlatformClient,
)


def create_gcp_cloud_platform_integration(
    *,
    client: GcpCloudPlatformIntegrationClient | None = None,
    enabled: bool = False,
) -> GcpCloudPlatformIntegration:
    """
    Build a contract-based GCP cloud platform integration.

    Compatibility shim — constructs Integration via from_store (create_gcp_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
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
