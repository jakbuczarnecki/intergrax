# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_selenium_browser_automation as _legacy_create_selenium_browser_automation

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.browser_automation.selenium.integration import (
    SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID,
    SeleniumBrowserAutomationIntegration,
    SeleniumBrowserAutomationIntegrationConfig,
    SeleniumBrowserAutomationClient,
)

__all__ = [
    "create_selenium_browser_automation",
    "create_selenium_browser_automation_integration",
]


def create_selenium_browser_automation_integration(
    *,
    client: SeleniumBrowserAutomationClient | None = None,
    enabled: bool = False,
) -> SeleniumBrowserAutomationIntegration:
    """
    Build a contract-based Selenium browser automation integration.

    The legacy facade (create_selenium_browser_automation) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Selenium browser automation integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SeleniumBrowserAutomationIntegration.from_client(client, enabled=enabled)
    return SeleniumBrowserAutomationIntegration.for_provider(
        provider_id=SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID,
        display_name="Selenium",
        config=SeleniumBrowserAutomationIntegrationConfig(enabled=enabled),
    )


def create_selenium_browser_automation(**kwargs: object) -> SeleniumBrowserAutomationIntegration:
    """Compatibility shim — constructs SeleniumBrowserAutomationIntegration from legacy runtime."""
    runtime = _legacy_create_selenium_browser_automation(**kwargs)
    if isinstance(runtime, SeleniumBrowserAutomationIntegration):
        return runtime
    return SeleniumBrowserAutomationIntegration.from_runtime(runtime)
