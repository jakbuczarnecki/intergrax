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


def navigation_timeout_enforced_for(browser: BrowserAutomation) -> bool:
    """Resolve timeout enforcement across integration facades and catalog clients."""
    seen: set[int] = set()
    current: object = browser
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BrowserAutomationNavigationTimeoutPort):
            return current.navigation_timeout_integration_enforced
        inner = getattr(current, "_client", None)
        if inner is None or inner is current:
            break
        current = inner
    return False


__all__ = [
    "BrowserAutomationNavigationTimeoutPort",
    "navigation_timeout_enforced_for",
]
