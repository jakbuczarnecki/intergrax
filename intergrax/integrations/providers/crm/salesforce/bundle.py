# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_salesforce_crm as _legacy_create_salesforce_crm

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.crm.salesforce.integration import (
    SALESFORCE_CRM_PROVIDER_ID,
    SalesforceCrmIntegration,
    SalesforceCrmIntegrationConfig,
    SalesforceCrmClient,
)

__all__ = [
    "create_salesforce_crm",
    "create_salesforce_crm_integration",
]


def create_salesforce_crm_integration(
    *,
    client: SalesforceCrmClient | None = None,
    enabled: bool = False,
) -> SalesforceCrmIntegration:
    """
    Build a contract-based Salesforce crm integration.

    The legacy facade (create_salesforce_crm) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Salesforce crm integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SalesforceCrmIntegration.from_client(client, enabled=enabled)
    return SalesforceCrmIntegration.for_provider(
        provider_id=SALESFORCE_CRM_PROVIDER_ID,
        display_name="Salesforce",
        config=SalesforceCrmIntegrationConfig(enabled=enabled),
    )


def create_salesforce_crm(**kwargs: object) -> SalesforceCrmIntegration:
    """Compatibility shim — constructs SalesforceCrmIntegration from legacy runtime."""
    runtime = _legacy_create_salesforce_crm(**kwargs)
    if isinstance(runtime, SalesforceCrmIntegration):
        return runtime
    return SalesforceCrmIntegration.from_client(runtime)
