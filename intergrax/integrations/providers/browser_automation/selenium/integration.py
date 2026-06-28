# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Selenium browser automation integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID = "selenium"


class SeleniumBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Selenium browser automation integration."""

    pass


@runtime_checkable
class SeleniumBrowserAutomationClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SeleniumBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Selenium browser automation integration.

    The legacy facade (create_selenium_browser_automation) remains separate and backward-compatible.
    """

    config: SeleniumBrowserAutomationIntegrationConfig = SeleniumBrowserAutomationIntegrationConfig()
    _client: SeleniumBrowserAutomationClient | None = PrivateAttr(default=None)

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
