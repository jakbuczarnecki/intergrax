# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete AWS integration bundle — the single composition root for AWS in Intergrax.

boto3 sessions are opened only in ``opens.py``. Tier-3 code MUST use
``create_aws_cloud_platform()``, ``create_aws_integration()``, or
``profile.resolve(IntegrationCategory.CLOUD_PLATFORM)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.cloud_platform.aws.config import AwsIntegrationConfig
from intergrax.integrations.providers.cloud_platform.aws.integration import (
    AWS_CLOUD_PLATFORM_PROVIDER_ID,
    AwsCloudPlatformClient,
    AwsCloudPlatformIntegration,
    AwsCloudPlatformIntegrationConfig,
)
from intergrax.integrations.providers.cloud_platform.aws.opens import open_aws_cloud_platform


@dataclass(frozen=True)
class AwsIntegrationBundle:
    config: AwsIntegrationConfig
    cloud_platform: AwsCloudPlatformIntegration


def resolve_aws_config(**overrides: object) -> AwsIntegrationConfig:
    return AwsIntegrationConfig.from_env(**overrides)


def create_aws_integration(
    *,
    cloud_platform: CloudPlatform | None = None,
    session: object | None = None,
    session_factory: Callable[[], object] | None = None,
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


def create_aws_cloud_platform_integration(
    *,
    client: AwsCloudPlatformClient | None = None,
    enabled: bool = False,
) -> AwsCloudPlatformIntegration:
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


def create_aws_cloud_platform(
    *,
    cloud_platform: CloudPlatform | None = None,
    session: object | None = None,
    session_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> AwsCloudPlatformIntegration:
    """Catalog factory for ``"aws"`` / ``CLOUD_PLATFORM``."""
    if cloud_platform is None and session is None and session_factory is None and not config_overrides:
        return create_aws_cloud_platform_integration(enabled=False)
    return create_aws_integration(
        cloud_platform=cloud_platform,
        session=session,
        session_factory=session_factory,
        **config_overrides,
    ).cloud_platform
