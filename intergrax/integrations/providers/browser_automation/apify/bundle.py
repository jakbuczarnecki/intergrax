# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_apify_browser_automation

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.browser_automation.apify.integration import (
    APIFY_BROWSER_AUTOMATION_PROVIDER_ID,
    ApifyBrowserAutomationIntegration,
    ApifyBrowserAutomationIntegrationConfig,
    ApifyBrowserAutomationClient,
)

__all__ = [
    "create_apify_browser_automation",
    "create_apify_browser_automation_integration",
]


def create_apify_browser_automation_integration(
    *,
    client: ApifyBrowserAutomationClient | None = None,
    enabled: bool = False,
) -> ApifyBrowserAutomationIntegration:
    """
    Build a contract-based Apify browser automation integration.

    The legacy facade (create_apify_browser_automation) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Apify browser automation integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ApifyBrowserAutomationIntegration.from_client(client, enabled=enabled)
    return ApifyBrowserAutomationIntegration.for_provider(
        provider_id=APIFY_BROWSER_AUTOMATION_PROVIDER_ID,
        display_name="Apify",
        config=ApifyBrowserAutomationIntegrationConfig(enabled=enabled),
    )
