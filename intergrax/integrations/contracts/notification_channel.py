# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Outbound notification contract — re-exports runtime adapter (§7.1.2, Phase M.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter


@runtime_checkable
class NotificationChannel(NotificationAdapter, Protocol):
    """Outbound notification channel with optional health probe."""

    def health(self) -> HealthStatus | bool: ...


__all__ = ["NotificationAdapter", "NotificationChannel"]
