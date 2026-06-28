# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID",
    "FirecrawlBrowserAutomationIntegration",
    "FirecrawlBrowserAutomationIntegrationConfig",
    "FirecrawlBrowserAutomationClient",
    "create_firecrawl_browser_automation",
    "create_firecrawl_browser_automation_integration",
    "register_firecrawl_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_firecrawl_browser_automation",
        "create_firecrawl_browser_automation_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID",
        "FirecrawlBrowserAutomationIntegration",
        "FirecrawlBrowserAutomationIntegrationConfig",
        "FirecrawlBrowserAutomationClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID",
        "FirecrawlBrowserAutomationIntegration",
        "FirecrawlBrowserAutomationIntegrationConfig",
        "FirecrawlBrowserAutomationClient",
    }
)

def __getattr__(name: str):
    if name == "register_firecrawl_integration":
        from intergrax.integrations.providers.browser_automation.firecrawl.register import register_firecrawl_integration

        return register_firecrawl_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.browser_automation.firecrawl import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.firecrawl import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.browser_automation.firecrawl import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
