# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integrations that honor per-navigation deadline configuration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BrowserAutomationNavigationTimeoutPort(Protocol):
    """Marker: ``fetch_page(..., navigation_timeout_s=...)`` is enforced by the integration."""


__all__ = ["BrowserAutomationNavigationTimeoutPort"]
