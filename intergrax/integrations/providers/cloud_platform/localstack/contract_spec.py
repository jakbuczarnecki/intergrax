# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Localstack cloud platform."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.localstack.bundle import (
    create_localstack_cloud_platform_integration,
)
from intergrax.integrations.providers.cloud_platform.localstack.integration import (
    LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID,
    LocalstackCloudPlatformIntegration,
    LocalstackCloudPlatformIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="cloud_platform",
    provider_id=LOCALSTACK_CLOUD_PLATFORM_PROVIDER_ID,
    integration_class=LocalstackCloudPlatformIntegration,
    contract_class=CloudPlatformIntegrationContract,
    contract_factory=create_localstack_cloud_platform_integration,
    display_name="Localstack",
    config_class=LocalstackCloudPlatformIntegrationConfig,
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
