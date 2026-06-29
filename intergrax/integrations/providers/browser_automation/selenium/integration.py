# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Selenium browser automation integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID = "selenium"


class SeleniumBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Selenium browser automation integration."""

    pass


SeleniumBrowserAutomationClient = BrowserAutomation

class SeleniumBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Single public Selenium browser automation entrypoint.

    Legacy catalog factory (create_selenium_browser_automation) owns catalog behavior; legacy factories use from_client().
    """

    config: SeleniumBrowserAutomationIntegrationConfig = SeleniumBrowserAutomationIntegrationConfig()
    _client: SeleniumBrowserAutomationClient | None = PrivateAttr(default=None)
    


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
        client: SeleniumBrowserAutomationClient,
        *,
        enabled: bool = False,
    ) -> SeleniumBrowserAutomationIntegration:
        integration = cls.for_provider(
            provider_id=SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID,
            display_name="Selenium",
            config=SeleniumBrowserAutomationIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SeleniumBrowserAutomationClient | None:
        return self._client

BrowserAutomation.register(SeleniumBrowserAutomationIntegration)
