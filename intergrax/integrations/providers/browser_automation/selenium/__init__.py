# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID",
    "SeleniumBrowserAutomationIntegration",
    "SeleniumBrowserAutomationIntegrationConfig",
    "SeleniumBrowserAutomationClient",
    "create_selenium_browser_automation",
    "create_selenium_browser_automation_integration",
    "register_selenium_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_selenium_browser_automation",
        "create_selenium_browser_automation_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID",
        "SeleniumBrowserAutomationIntegration",
        "SeleniumBrowserAutomationIntegrationConfig",
        "SeleniumBrowserAutomationClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID",
        "SeleniumBrowserAutomationIntegration",
        "SeleniumBrowserAutomationIntegrationConfig",
        "SeleniumBrowserAutomationClient",
    }
)

def __getattr__(name: str):
    if name == "register_selenium_integration":
        from intergrax.integrations.providers.browser_automation.selenium.register import register_selenium_integration

        return register_selenium_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.browser_automation.selenium import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.selenium import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.selenium import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
