# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete AWS integration bundle — the single composition root for AWS in Intergrax.

boto3 sessions are opened only in ``opens.py``. Tier-3 code MUST use
``create_aws_cloud_platform()``, ``create_aws_integration()``, or
``profile.resolve(IntegrationCategory.CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.cloud_platform.aws.adapter import _AwsCloudPlatform
from intergrax.integrations.providers.cloud_platform.aws.config import AwsIntegrationConfig
from intergrax.integrations.providers.cloud_platform.aws.opens import open_aws_cloud_platform


@dataclass(frozen=True)
class AwsIntegrationBundle:
    config: AwsIntegrationConfig
    cloud_platform: AwsCloudPlatformIntegration


def resolve_aws_config(**overrides: object) -> AwsIntegrationConfig:
    return AwsIntegrationConfig.from_env(**overrides)


def create_aws_integration(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> AwsIntegrationBundle:
    config = resolve_aws_config(**config_overrides)
    platform = open_aws_cloud_platform(
        config,
        implementation=cloud_platform,
        session=session,
        session_factory=session_factory,
    )
    assert isinstance(platform, AwsCloudPlatformIntegration)
    return AwsIntegrationBundle(config=config, cloud_platform=platform)


def create_aws_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> AwsCloudPlatformIntegration:
    """Catalog factory for ``"aws"`` / ``CLOUD_PLATFORM``."""
    return create_aws_integration(
        cloud_platform=cloud_platform,
        session=session,
        session_factory=session_factory,
        **config_overrides,
    ).cloud_platform

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.cloud_platform.aws.integration import (
    AWS_CLOUD_PLATFORM_PROVIDER_ID,
    AwsCloudPlatformIntegration,
    AwsCloudPlatformIntegrationConfig,
    AwsCloudPlatformClient,
)


def create_aws_cloud_platform_integration(
    *,
    client: AwsCloudPlatformIntegrationClient | None = None,
    enabled: bool = False,
) -> AwsCloudPlatformIntegration:
    """
    Build a contract-based AWS cloud platform integration.

    Compatibility shim — constructs Integration via from_store (create_aws_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "AWS cloud platform integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AwsCloudPlatformIntegration.from_client(client, enabled=enabled)
    return AwsCloudPlatformIntegration.for_provider(
        provider_id=AWS_CLOUD_PLATFORM_PROVIDER_ID,
        display_name="AWS",
        config=AwsCloudPlatformIntegrationConfig(enabled=enabled),
    )
