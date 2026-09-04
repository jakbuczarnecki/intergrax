# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Serpapi."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.serpapi.bundle import (
    create_serpapi_search_provider_integration,
)
from intergrax.integrations.providers.search_provider.serpapi.integration import (
    SERPAPI_SEARCH_PROVIDER_PROVIDER_ID,
    SerpapiSearchProviderIntegration,
    SerpapiSearchProviderIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="search_provider",
    provider_id=SERPAPI_SEARCH_PROVIDER_PROVIDER_ID,
    integration_class=SerpapiSearchProviderIntegration,
    contract_class=SearchProviderIntegrationContract,
    contract_factory=create_serpapi_search_provider_integration,
    display_name="Serpapi",
    config_class=SerpapiSearchProviderIntegrationConfig,
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
