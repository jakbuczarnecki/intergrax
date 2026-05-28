# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Notification adapter contracts (§18)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.notifications.models import NotificationMessage


@runtime_checkable
class NotificationAdapter(Protocol):
    """Surface-facing adapter: canonical message in, channel-specific delivery out."""

    async def notify(self, message: NotificationMessage) -> None: ...
