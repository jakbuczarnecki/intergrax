# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Outbound notification adapter contract (integration catalog surface)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.notification_message import NotificationMessage


@runtime_checkable
class NotificationAdapter(Protocol):
    """Surface-facing adapter: canonical message in, channel-specific delivery out."""

    async def notify(self, message: NotificationMessage) -> None: ...


__all__ = ["NotificationAdapter", "NotificationMessage"]
