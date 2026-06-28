# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Browserbase browser automation integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BROWSERBASE_BROWSER_AUTOMATION_PROVIDER_ID = "browserbase"


class BrowserbaseBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Browserbase browser automation integration."""

    pass


@runtime_checkable
class BrowserbaseBrowserAutomationClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BrowserbaseBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Browserbase browser automation integration.

    The legacy facade (create_browserbase_browser_automation) remains separate and backward-compatible.
    """

    config: BrowserbaseBrowserAutomationIntegrationConfig = BrowserbaseBrowserAutomationIntegrationConfig()
    _client: BrowserbaseBrowserAutomationClient | None = PrivateAttr(default=None)

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
