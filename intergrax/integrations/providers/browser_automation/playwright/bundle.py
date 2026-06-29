# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_playwright_browser_automation as _legacy_create_playwright_browser_automation

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.browser_automation.playwright.integration import (
    PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID,
    PlaywrightBrowserAutomationIntegration,
    PlaywrightBrowserAutomationIntegrationConfig,
    PlaywrightBrowserAutomationClient,
)

__all__ = [
    "create_playwright_browser_automation",
    "create_playwright_browser_automation_integration",
]


def create_playwright_browser_automation_integration(
    *,
    client: PlaywrightBrowserAutomationClient | None = None,
    enabled: bool = False,
) -> PlaywrightBrowserAutomationIntegration:
    """
    Build a contract-based Playwright browser automation integration.

    The legacy facade (create_playwright_browser_automation) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Playwright browser automation integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PlaywrightBrowserAutomationIntegration.from_client(client, enabled=enabled)
    return PlaywrightBrowserAutomationIntegration.for_provider(
        provider_id=PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID,
        display_name="Playwright",
        config=PlaywrightBrowserAutomationIntegrationConfig(enabled=enabled),
    )


def create_playwright_browser_automation(**kwargs: object) -> PlaywrightBrowserAutomationIntegration:
    """Compatibility shim — constructs PlaywrightBrowserAutomationIntegration from legacy runtime."""
    runtime = _legacy_create_playwright_browser_automation(**kwargs)
    if isinstance(runtime, PlaywrightBrowserAutomationIntegration):
        return runtime
    return PlaywrightBrowserAutomationIntegration.from_client(runtime)
