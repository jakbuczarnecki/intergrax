# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Browserbase browser automation integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.integrations.contracts.browser_automation_navigation import (
    navigation_timeout_enforced_for_catalog_client,
)
from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BROWSERBASE_BROWSER_AUTOMATION_PROVIDER_ID = "browserbase"


class BrowserbaseBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Browserbase browser automation integration."""

    pass


BrowserbaseBrowserAutomationClient = BrowserAutomation

class BrowserbaseBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Single public Browserbase browser automation entrypoint.

    Legacy catalog factory (create_browserbase_browser_automation) owns catalog behavior; legacy factories use from_client().
    """

    config: BrowserbaseBrowserAutomationIntegrationConfig = BrowserbaseBrowserAutomationIntegrationConfig()
    _client: BrowserbaseBrowserAutomationClient | None = PrivateAttr(default=None)
    


    def fetch_page(
        self,
        url: str,
        *,
        wait_until: str = "load",
        navigation_timeout_s: float | None = None,
    ) -> PageContent:
        return self._require_client().fetch_page(
            url,
            wait_until=wait_until,
            navigation_timeout_s=navigation_timeout_s,
        )

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
        client: BrowserbaseBrowserAutomationClient,
        *,
        enabled: bool = False,
    ) -> BrowserbaseBrowserAutomationIntegration:
        integration = cls.for_provider(
            provider_id=BROWSERBASE_BROWSER_AUTOMATION_PROVIDER_ID,
            display_name="Browserbase",
            config=BrowserbaseBrowserAutomationIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BrowserbaseBrowserAutomationClient | None:
        return self._client

    @property
    def navigation_timeout_integration_enforced(self) -> bool:
        return navigation_timeout_enforced_for_catalog_client(self.client)

BrowserAutomation.register(BrowserbaseBrowserAutomationIntegration)
