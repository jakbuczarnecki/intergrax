# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Outbound notification transport contracts (§18, Phase H.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class NotificationDelivery(ABC):
    """
    Transport for outbound notification payloads.

    Implementations: HTTP webhook, queue publisher, email SMTP, etc.
    Adapters depend on this contract — not on httpx or a specific vendor.
    """

    @abstractmethod
    async def deliver(
        self,
        *,
        destination: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Deliver ``payload`` to ``destination`` (URL, topic, mailbox, …)."""


class NullNotificationDelivery(NotificationDelivery):
    """Explicit no-op transport for tests and disabled delivery."""

    async def deliver(
        self,
        *,
        destination: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        _ = destination, payload, headers
