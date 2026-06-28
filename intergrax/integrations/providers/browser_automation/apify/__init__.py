# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "APIFY_BROWSER_AUTOMATION_PROVIDER_ID",
    "ApifyBrowserAutomationIntegration",
    "ApifyBrowserAutomationIntegrationConfig",
    "ApifyBrowserAutomationClient",
    "create_apify_browser_automation",
    "create_apify_browser_automation_integration",
    "register_apify_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_apify_browser_automation",
        "create_apify_browser_automation_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "APIFY_BROWSER_AUTOMATION_PROVIDER_ID",
        "ApifyBrowserAutomationIntegration",
        "ApifyBrowserAutomationIntegrationConfig",
        "ApifyBrowserAutomationClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "APIFY_BROWSER_AUTOMATION_PROVIDER_ID",
        "ApifyBrowserAutomationIntegration",
        "ApifyBrowserAutomationIntegrationConfig",
        "ApifyBrowserAutomationClient",
    }
)

def __getattr__(name: str):
    if name == "register_apify_integration":
        from intergrax.integrations.providers.browser_automation.apify.register import register_apify_integration

        return register_apify_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.browser_automation.apify import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.apify import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.apify import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
