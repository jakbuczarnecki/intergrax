# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AWS cloud platform integration (Phase M.6)."""

from intergrax.utils.lazy_export import export_from_bundle
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
    "create_aws_cloud_platform_integration",
]

_LAZY_EXPORTS = frozenset(
    {
        "AwsIntegrationBundle",
        "AwsCloudPlatform",
        "create_aws_integration",
        "create_aws_cloud_platform",
        "register_aws_integration",
        "resolve_aws_config",
        "create_aws_cloud_platform_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AWS_CLOUD_PLATFORM_PROVIDER_ID",
        "AwsCloudPlatformIntegration",
        "AwsCloudPlatformIntegrationConfig",
        "AwsCloudPlatformClient",
    }
)

def __getattr__(name: str):
    if name == "register_aws_integration":
        from intergrax.integrations.providers.cloud_platform.aws.register import register_aws_integration

        return register_aws_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.aws import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "AwsCloudPlatform":
        from intergrax.integrations.providers.cloud_platform.aws.adapter import _AwsCloudPlatform

        return AwsCloudPlatform
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.cloud_platform.aws import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
