# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Firecrawl browser automation integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID = "firecrawl"


class FirecrawlBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Firecrawl browser automation integration."""

    pass


@runtime_checkable
class FirecrawlBrowserAutomationClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class FirecrawlBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Firecrawl browser automation integration.

    The legacy facade (create_firecrawl_browser_automation) remains separate and backward-compatible.
    """

    config: FirecrawlBrowserAutomationIntegrationConfig = FirecrawlBrowserAutomationIntegrationConfig()
    _client: FirecrawlBrowserAutomationClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: FirecrawlBrowserAutomationClient,
        *,
        enabled: bool = False,
    ) -> FirecrawlBrowserAutomationIntegration:
        integration = cls.for_provider(
            provider_id=FIRECRAWL_BROWSER_AUTOMATION_PROVIDER_ID,
            display_name="Firecrawl",
            config=FirecrawlBrowserAutomationIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> FirecrawlBrowserAutomationClient | None:
        return self._client
