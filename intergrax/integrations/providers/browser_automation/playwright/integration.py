# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Playwright browser automation integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PLAYWRIGHT_BROWSER_AUTOMATION_PROVIDER_ID = "playwright"


class PlaywrightBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Playwright browser automation integration."""

    pass


@runtime_checkable
class PlaywrightBrowserAutomationClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PlaywrightBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Playwright browser automation integration.

    The legacy facade (create_playwright_browser_automation) remains separate and backward-compatible.
    """

    config: PlaywrightBrowserAutomationIntegrationConfig = PlaywrightBrowserAutomationIntegrationConfig()
    _client: PlaywrightBrowserAutomationClient | None = PrivateAttr(default=None)

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
