# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Outbound notification contract (integration catalog surface)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.notification_adapter import NotificationAdapter


@runtime_checkable
class NotificationChannel(NotificationAdapter, Protocol):
    """Outbound notification channel with optional health probe."""

    def health(self) -> HealthStatus | bool: ...


__all__ = ["NotificationAdapter", "NotificationChannel"]
