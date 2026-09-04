# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Salesforce."""

from __future__ import annotations

from intergrax.integrations.providers.crm.salesforce.bundle import (
    create_salesforce_crm_integration,
)
from intergrax.integrations.providers.crm.salesforce.integration import (
    SALESFORCE_CRM_PROVIDER_ID,
    SalesforceCrmIntegration,
    SalesforceCrmIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.automation import CrmIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="crm",
    provider_id=SALESFORCE_CRM_PROVIDER_ID,
    integration_class=SalesforceCrmIntegration,
    contract_class=CrmIntegrationContract,
    contract_factory=create_salesforce_crm_integration,
    display_name="Salesforce",
    config_class=SalesforceCrmIntegrationConfig,
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
