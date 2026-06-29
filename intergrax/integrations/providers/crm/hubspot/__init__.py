# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "HUBSPOT_CRM_PROVIDER_ID",
    "HubspotCrmIntegration",
    "HubspotCrmIntegrationConfig",
    "HubspotCrmClient",
    "create_hubspot_crm",
    "create_hubspot_crm_integration",
    "register_hubspot_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_hubspot_crm",
        "create_hubspot_crm_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "HUBSPOT_CRM_PROVIDER_ID",
        "HubspotCrmIntegration",
        "HubspotCrmIntegrationConfig",
        "HubspotCrmClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "HUBSPOT_CRM_PROVIDER_ID",
        "HubspotCrmIntegration",
        "HubspotCrmIntegrationConfig",
        "HubspotCrmClient",
    }
)

def __getattr__(name: str):
    if name == "register_hubspot_integration":
        from intergrax.integrations.providers.crm.hubspot.register import register_hubspot_integration

        return register_hubspot_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.crm.hubspot import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.crm.hubspot import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.crm.hubspot import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
