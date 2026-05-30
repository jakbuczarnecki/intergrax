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
from intergrax.integrations.providers.aws.adapter import AwsCloudPlatform
from intergrax.integrations.providers.aws.config import AwsIntegrationConfig
from intergrax.integrations.providers.aws.opens import open_aws_cloud_platform


@dataclass(frozen=True)
class AwsIntegrationBundle:
    config: AwsIntegrationConfig
    cloud_platform: AwsCloudPlatform


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
    assert isinstance(platform, AwsCloudPlatform)
    return AwsIntegrationBundle(config=config, cloud_platform=platform)


def create_aws_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    session: Optional[object] = None,
    session_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> AwsCloudPlatform:
    """Catalog factory for ``IntegrationSlug.AWS`` / ``CLOUD_PLATFORM``."""
    return create_aws_integration(
        cloud_platform=cloud_platform,
        session=session,
        session_factory=session_factory,
        **config_overrides,
    ).cloud_platform
