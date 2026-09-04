# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Hubspot."""

from __future__ import annotations

from intergrax.integrations.providers.crm.hubspot.bundle import (
    create_hubspot_crm_integration,
)
from intergrax.integrations.providers.crm.hubspot.integration import (
    HUBSPOT_CRM_PROVIDER_ID,
    HubspotCrmIntegration,
    HubspotCrmIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.automation import CrmIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="crm",
    provider_id=HUBSPOT_CRM_PROVIDER_ID,
    integration_class=HubspotCrmIntegration,
    contract_class=CrmIntegrationContract,
    contract_factory=create_hubspot_crm_integration,
    display_name="Hubspot",
    config_class=HubspotCrmIntegrationConfig,
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
