# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_hubspot_crm

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.crm.hubspot.integration import (
    HUBSPOT_CRM_PROVIDER_ID,
    HubspotCrmIntegration,
    HubspotCrmIntegrationConfig,
    HubspotCrmClient,
)

__all__ = [
    "create_hubspot_crm",
    "create_hubspot_crm_integration",
]


def create_hubspot_crm_integration(
    *,
    client: HubspotCrmClient | None = None,
    enabled: bool = False,
) -> HubspotCrmIntegration:
    """
    Build a contract-based Hubspot crm integration.

    The legacy facade (create_hubspot_crm) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Hubspot crm integration requires an injected client when enabled=True",
        )
    if client is not None:
        return HubspotCrmIntegration.from_client(client, enabled=enabled)
    return HubspotCrmIntegration.for_provider(
        provider_id=HUBSPOT_CRM_PROVIDER_ID,
        display_name="Hubspot",
        config=HubspotCrmIntegrationConfig(enabled=enabled),
    )
