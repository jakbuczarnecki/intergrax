# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_firecrawl_browser_automation as _legacy_create_firecrawl_browser_automation

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.browser_automation.firecrawl.integration import (
    FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID,
    FirecrawlBrowserAutomationIntegration,
    FirecrawlBrowserAutomationIntegrationConfig,
    FirecrawlBrowserAutomationClient,
)

__all__ = [
    "create_firecrawl_browser_automation",
    "create_firecrawl_browser_automation_integration",
]


def create_firecrawl_browser_automation_integration(
    *,
    client: FirecrawlBrowserAutomationClient | None = None,
    enabled: bool = False,
) -> FirecrawlBrowserAutomationIntegration:
    """
    Build a contract-based Firecrawl browser automation integration.

    The legacy facade (create_firecrawl_browser_automation) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Firecrawl browser automation integration requires an injected client when enabled=True",
        )
    if client is not None:
        return FirecrawlBrowserAutomationIntegration.from_client(client, enabled=enabled)
    return FirecrawlBrowserAutomationIntegration.for_provider(
        provider_id=FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID,
        display_name="Firecrawl",
        config=FirecrawlBrowserAutomationIntegrationConfig(enabled=enabled),
    )


def create_firecrawl_browser_automation(**kwargs: object) -> FirecrawlBrowserAutomationIntegration:
    """Compatibility shim — constructs FirecrawlBrowserAutomationIntegration from legacy runtime."""
    runtime = _legacy_create_firecrawl_browser_automation(**kwargs)
    if isinstance(runtime, FirecrawlBrowserAutomationIntegration):
        return runtime
    return FirecrawlBrowserAutomationIntegration.from_runtime(runtime)
