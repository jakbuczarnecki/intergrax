# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AWS cloud platform integration (Phase M.6)."""

from intergrax.integrations.providers.cloud_platform.aws.config import (
    ENV_AWS_PROFILE,
    ENV_AWS_REGION,
    ENV_AWS_ROLE_ARN,
    AwsIntegrationConfig,
)

__all__ = [
    "ENV_AWS_PROFILE",
    "ENV_AWS_REGION",
    "ENV_AWS_ROLE_ARN",
    "AwsCloudPlatform",
    "AwsIntegrationBundle",
    "AwsIntegrationConfig",
    "create_aws_cloud_platform",
    "create_aws_integration",
    "register_aws_integration",
    "resolve_aws_config",
]

_LAZY_EXPORTS = frozenset(
    {
        "AwsIntegrationBundle",
        "AwsCloudPlatform",
        "create_aws_integration",
        "create_aws_cloud_platform",
        "register_aws_integration",
        "resolve_aws_config",
    }
)


def __getattr__(name: str):
    if name == "register_aws_integration":
        from intergrax.integrations.providers.cloud_platform.aws.register import register_aws_integration

        return register_aws_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.aws import bundle as _bundle

        return getattr(_bundle, name)
    if name == "AwsCloudPlatform":
        from intergrax.integrations.providers.cloud_platform.aws.adapter import AwsCloudPlatform

        return AwsCloudPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
