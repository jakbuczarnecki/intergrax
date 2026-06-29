# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Playwright browser automation integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID = "playwright"


class PlaywrightBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Playwright browser automation integration."""

    pass


PlaywrightBrowserAutomationClient = BrowserAutomation

class PlaywrightBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Single public Playwright browser automation entrypoint.

    Legacy catalog factory (create_playwright_browser_automation) owns catalog behavior; legacy factories use from_client().
    """

    config: PlaywrightBrowserAutomationIntegrationConfig = PlaywrightBrowserAutomationIntegrationConfig()
    _client: PlaywrightBrowserAutomationClient | None = PrivateAttr(default=None)
    


    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        return self._require_client().fetch_page(url, wait_until=wait_until)

    def close(self) -> None:
        self._require_client().close()


    def _require_client(self) -> BrowserAutomation:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: PlaywrightBrowserAutomationClient,
        *,
        enabled: bool = False,
    ) -> PlaywrightBrowserAutomationIntegration:
        integration = cls.for_provider(
            provider_id=PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID,
            display_name="Playwright",
            config=PlaywrightBrowserAutomationIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PlaywrightBrowserAutomationClient | None:
        return self._client

BrowserAutomation.register(PlaywrightBrowserAutomationIntegration)
