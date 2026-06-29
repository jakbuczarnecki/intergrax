# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SALESFORCE_CRM_PROVIDER_ID",
    "SalesforceCrmIntegration",
    "SalesforceCrmIntegrationConfig",
    "SalesforceCrmClient",
    "create_salesforce_crm",
    "create_salesforce_crm_integration",
    "register_salesforce_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_salesforce_crm",
        "create_salesforce_crm_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SALESFORCE_CRM_PROVIDER_ID",
        "SalesforceCrmIntegration",
        "SalesforceCrmIntegrationConfig",
        "SalesforceCrmClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SALESFORCE_CRM_PROVIDER_ID",
        "SalesforceCrmIntegration",
        "SalesforceCrmIntegrationConfig",
        "SalesforceCrmClient",
    }
)

def __getattr__(name: str):
    if name == "register_salesforce_integration":
        from intergrax.integrations.providers.crm.salesforce.register import register_salesforce_integration

        return register_salesforce_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.crm.salesforce import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.crm.salesforce import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.crm.salesforce import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
