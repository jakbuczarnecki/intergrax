# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Clerk identity provider."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.clerk.bundle import (
    create_clerk_identity_provider_integration,
)
from intergrax.integrations.providers.identity_provider.clerk.integration import (
    CLERK_IDENTITY_PROVIDER_PROVIDER_ID,
    ClerkIdentityProviderIntegration,
    ClerkIdentityProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="identity_provider",
    provider_id=CLERK_IDENTITY_PROVIDER_PROVIDER_ID,
    integration_class=ClerkIdentityProviderIntegration,
    contract_class=IdentityProviderIntegrationContract,
    contract_factory=create_clerk_identity_provider_integration,
    display_name="Clerk",
    config_class=ClerkIdentityProviderIntegrationConfig,
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
