# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Daytona sandbox host."""

from __future__ import annotations

from intergrax.integrations.providers.sandbox_host.daytona.bundle import (
    create_daytona_sandbox_host_integration,
)
from intergrax.integrations.providers.sandbox_host.daytona.integration import (
    DAYTONA_SANDBOX_HOST_PROVIDER_ID,
    DaytonaSandboxHostIntegration,
    DaytonaSandboxHostIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="sandbox_host",
    provider_id=DAYTONA_SANDBOX_HOST_PROVIDER_ID,
    integration_class=DaytonaSandboxHostIntegration,
    contract_class=SandboxHostIntegrationContract,
    contract_factory=create_daytona_sandbox_host_integration,
    display_name="Daytona",
    config_class=DaytonaSandboxHostIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
