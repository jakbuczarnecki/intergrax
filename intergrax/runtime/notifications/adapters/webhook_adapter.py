# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Generic webhook notification adapter (Phase H.1).

Composes a payload formatter + delivery transport — reusable for Slack, Teams,
or any HTTP webhook without coupling Nexus to a single vendor SDK.
"""

from __future__ import annotations

import logging
from typing import Optional

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.deliveries.http_webhook_delivery import HttpWebhookDelivery
from intergrax.runtime.notifications.formatters import NotificationPayloadFormatter

logger = logging.getLogger(__name__)


class WebhookNotificationAdapter:
    """
    Delivers ``NotificationMessage`` via injectable transport + formatter.

    When ``webhook_url`` is empty the adapter is a no-op (safe default).
    Delivery errors are logged and swallowed (best-effort §42.26 / §18).
    """

    def __init__(
        self,
        *,
        webhook_url: str,
        formatter: NotificationPayloadFormatter,
        delivery: Optional[NotificationDelivery] = None,
        channel: str = "webhook",
        fail_silently: bool = True,
    ) -> None:
        self._webhook_url = webhook_url.strip()
        self._formatter = formatter
        self._delivery = delivery or HttpWebhookDelivery()
        self._channel = channel
        self._fail_silently = fail_silently

    @property
    def webhook_url(self) -> str:
        return self._webhook_url

    async def notify(self, message: NotificationMessage) -> None:
        if not self._webhook_url:
            return
        payload = self._formatter.format(message)
        try:
            await self._delivery.deliver(destination=self._webhook_url, payload=payload)
        except Exception as exc:
            if self._fail_silently:
                logger.warning(
                    "notification delivery failed channel=%s task=%s error=%s",
                    self._channel,
                    message.task_id,
                    exc,
                )
                return
            raise
