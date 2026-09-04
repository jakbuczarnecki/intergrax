# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Workos identity provider."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.workos.bundle import (
    create_workos_identity_provider_integration,
)
from intergrax.integrations.providers.identity_provider.workos.integration import (
    WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
    WorkosIdentityProviderIntegration,
    WorkosIdentityProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="identity_provider",
    provider_id=WORKOS_IDENTITY_PROVIDER_PROVIDER_ID,
    integration_class=WorkosIdentityProviderIntegration,
    contract_class=IdentityProviderIntegrationContract,
    contract_factory=create_workos_identity_provider_integration,
    display_name="Workos",
    config_class=WorkosIdentityProviderIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
