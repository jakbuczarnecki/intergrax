# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for AWS cloud platform."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.aws.bundle import (
    create_aws_cloud_platform_integration,
)
from intergrax.integrations.providers.cloud_platform.aws.integration import (
    AWS_CLOUD_PLATFORM_PROVIDER_ID,
    AwsCloudPlatformIntegration,
    AwsCloudPlatformIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="cloud_platform",
    provider_id=AWS_CLOUD_PLATFORM_PROVIDER_ID,
    integration_class=AwsCloudPlatformIntegration,
    contract_class=CloudPlatformIntegrationContract,
    contract_factory=create_aws_cloud_platform_integration,
    display_name="AWS",
    config_class=AwsCloudPlatformIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
