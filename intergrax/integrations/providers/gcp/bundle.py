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
from intergrax.integrations.providers.gcp.adapter import GcpCloudPlatform
from intergrax.integrations.providers.gcp.config import GcpIntegrationConfig
from intergrax.integrations.providers.gcp.opens import open_gcp_cloud_platform


@dataclass(frozen=True)
class GcpIntegrationBundle:
    config: GcpIntegrationConfig
    cloud_platform: GcpCloudPlatform


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
    assert isinstance(platform, GcpCloudPlatform)
    return GcpIntegrationBundle(config=config, cloud_platform=platform)


def create_gcp_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    credentials: Optional[object] = None,
    resolved_project_id: str = "",
    credential_factory: Optional[Callable[[], tuple[object, str]]] = None,
    **config_overrides: object,
) -> GcpCloudPlatform:
    """Catalog factory for ``IntegrationSlug.GCP`` / ``CLOUD_PLATFORM``."""
    return create_gcp_integration(
        cloud_platform=cloud_platform,
        credentials=credentials,
        resolved_project_id=resolved_project_id,
        credential_factory=credential_factory,
        **config_overrides,
    ).cloud_platform
