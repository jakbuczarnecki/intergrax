# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID",
    "PlaywrightBrowserAutomationIntegration",
    "PlaywrightBrowserAutomationIntegrationConfig",
    "PlaywrightBrowserAutomationClient",
    "create_playwright_browser_automation",
    "create_playwright_browser_automation_integration",
    "register_playwright_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_playwright_browser_automation",
        "create_playwright_browser_automation_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID",
        "PlaywrightBrowserAutomationIntegration",
        "PlaywrightBrowserAutomationIntegrationConfig",
        "PlaywrightBrowserAutomationClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID",
        "PlaywrightBrowserAutomationIntegration",
        "PlaywrightBrowserAutomationIntegrationConfig",
        "PlaywrightBrowserAutomationClient",
    }
)

def __getattr__(name: str):
    if name == "register_playwright_integration":
        from intergrax.integrations.providers.browser_automation.playwright.register import register_playwright_integration

        return register_playwright_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.browser_automation.playwright import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.playwright import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.playwright import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
