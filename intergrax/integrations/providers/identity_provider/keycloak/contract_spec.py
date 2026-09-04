# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Keycloak identity provider."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.keycloak.bundle import (
    create_keycloak_identity_provider_integration,
)
from intergrax.integrations.providers.identity_provider.keycloak.integration import (
    KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID,
    KeycloakIdentityProviderIntegration,
    KeycloakIdentityProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="identity_provider",
    provider_id=KEYCLOAK_IDENTITY_PROVIDER_PROVIDER_ID,
    integration_class=KeycloakIdentityProviderIntegration,
    contract_class=IdentityProviderIntegrationContract,
    contract_factory=create_keycloak_identity_provider_integration,
    display_name="Keycloak",
    config_class=KeycloakIdentityProviderIntegrationConfig,
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
