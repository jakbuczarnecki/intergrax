# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Google Cse."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.google_cse.bundle import (
    create_google_cse_search_provider_integration,
)
from intergrax.integrations.providers.search_provider.google_cse.integration import (
    GOOGLE_CSE_SEARCH_PROVIDER_PROVIDER_ID,
    GoogleCseSearchProviderIntegration,
    GoogleCseSearchProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="search_provider",
    provider_id=GOOGLE_CSE_SEARCH_PROVIDER_PROVIDER_ID,
    integration_class=GoogleCseSearchProviderIntegration,
    contract_class=SearchProviderIntegrationContract,
    contract_factory=create_google_cse_search_provider_integration,
    display_name="Google Cse",
    config_class=GoogleCseSearchProviderIntegrationConfig,
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
