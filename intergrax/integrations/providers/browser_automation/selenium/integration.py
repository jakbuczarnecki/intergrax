# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Selenium browser automation integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
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
    Single public Selenium browser automation entrypoint.

    Legacy catalog factory (create_selenium_browser_automation) delegates to this class.
    """

    config: SeleniumBrowserAutomationIntegrationConfig = SeleniumBrowserAutomationIntegrationConfig()
    _client: SeleniumBrowserAutomationClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> SeleniumBrowserAutomationIntegration:
        integration = cls.for_provider(
            provider_id=SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID,
            display_name="Selenium",
            config=SeleniumBrowserAutomationIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        return self._require_runtime().fetch_page(url, wait_until=wait_until)

    def close(self) -> None:
        self._require_runtime().close()


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

BrowserAutomation.register(SeleniumBrowserAutomationIntegration)
