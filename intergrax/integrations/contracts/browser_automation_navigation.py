# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integrations that honor per-navigation deadline configuration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.browser_automation import BrowserAutomation


@runtime_checkable
class BrowserAutomationNavigationTimeoutPort(Protocol):
    """
    Positive capability: ``navigation_timeout_integration_enforced`` is True only when
    ``fetch_page(..., navigation_timeout_s=...)`` is honored by the physical integration.
    """

    @property
    def navigation_timeout_integration_enforced(self) -> bool:
        """Whether ``navigation_timeout_s`` is enforced for navigation on this integration."""


def navigation_timeout_enforced_for_catalog_client(
    client: BrowserAutomation | None,
) -> bool:
    """Positive capability from a catalog ``BrowserAutomation`` client (public delegation)."""
    if client is None:
        return False
    if isinstance(client, BrowserAutomationNavigationTimeoutPort):
        return client.navigation_timeout_integration_enforced
    return False


def navigation_timeout_enforced_for(browser: BrowserAutomation) -> bool:
    """Resolve timeout enforcement via sanctioned ``BrowserAutomationNavigationTimeoutPort``."""
    if isinstance(browser, BrowserAutomationNavigationTimeoutPort):
        return browser.navigation_timeout_integration_enforced
    return False


__all__ = [
    "BrowserAutomationNavigationTimeoutPort",
    "navigation_timeout_enforced_for",
    "navigation_timeout_enforced_for_catalog_client",
]
