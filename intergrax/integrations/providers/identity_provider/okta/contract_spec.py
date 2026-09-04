# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Okta identity provider."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.okta.bundle import (
    create_okta_identity_provider_integration,
)
from intergrax.integrations.providers.identity_provider.okta.integration import (
    OKTA_IDENTITY_PROVIDER_PROVIDER_ID,
    OktaIdentityProviderIntegration,
    OktaIdentityProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import IdentityProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="identity_provider",
    provider_id=OKTA_IDENTITY_PROVIDER_PROVIDER_ID,
    integration_class=OktaIdentityProviderIntegration,
    contract_class=IdentityProviderIntegrationContract,
    contract_factory=create_okta_identity_provider_integration,
    display_name="Okta",
    config_class=OktaIdentityProviderIntegrationConfig,
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
