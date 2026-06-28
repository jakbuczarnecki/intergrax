# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Apify browser automation integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

APIFY_BROWSER_AUTOMATION_PROVIDER_ID = "apify"


class ApifyBrowserAutomationIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Apify browser automation integration."""

    pass


@runtime_checkable
class ApifyBrowserAutomationClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ApifyBrowserAutomationIntegration(BrowserAutomationIntegrationContract):
    """
    Apify browser automation integration.

    The legacy facade (create_apify_browser_automation) remains separate and backward-compatible.
    """

    config: ApifyBrowserAutomationIntegrationConfig = ApifyBrowserAutomationIntegrationConfig()
    _client: ApifyBrowserAutomationClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ApifyBrowserAutomationClient,
        *,
        enabled: bool = False,
    ) -> ApifyBrowserAutomationIntegration:
        integration = cls.for_provider(
            provider_id=APIFY_BROWSER_AUTOMATION_PROVIDER_ID,
            display_name="Apify",
            config=ApifyBrowserAutomationIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ApifyBrowserAutomationClient | None:
        return self._client
