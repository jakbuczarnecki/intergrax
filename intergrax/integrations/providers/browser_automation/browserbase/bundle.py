# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_browserbase_browser_automation as _legacy_create_browserbase_browser_automation

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.browser_automation.browserbase.integration import (
    BROWSERBASE_BROWSER_AUTOMATION_PROVIDER_ID,
    BrowserbaseBrowserAutomationIntegration,
    BrowserbaseBrowserAutomationIntegrationConfig,
    BrowserbaseBrowserAutomationClient,
)

__all__ = [
    "create_browserbase_browser_automation",
    "create_browserbase_browser_automation_integration",
]


def create_browserbase_browser_automation_integration(
    *,
    client: BrowserbaseBrowserAutomationClient | None = None,
    enabled: bool = False,
) -> BrowserbaseBrowserAutomationIntegration:
    """
    Build a contract-based Browserbase browser automation integration.

    The legacy facade (create_browserbase_browser_automation) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Browserbase browser automation integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BrowserbaseBrowserAutomationIntegration.from_client(client, enabled=enabled)
    return BrowserbaseBrowserAutomationIntegration.for_provider(
        provider_id=BROWSERBASE_BROWSER_AUTOMATION_PROVIDER_ID,
        display_name="Browserbase",
        config=BrowserbaseBrowserAutomationIntegrationConfig(enabled=enabled),
    )


def create_browserbase_browser_automation(**kwargs: object) -> BrowserbaseBrowserAutomationIntegration:
    """Compatibility shim — constructs BrowserbaseBrowserAutomationIntegration from legacy runtime."""
    runtime = _legacy_create_browserbase_browser_automation(**kwargs)
    if isinstance(runtime, BrowserbaseBrowserAutomationIntegration):
        return runtime
    return BrowserbaseBrowserAutomationIntegration.from_client(runtime)
