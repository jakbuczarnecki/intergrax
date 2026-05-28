# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Laboratory logging adapter — no external I/O."""

from __future__ import annotations

import logging

from intergrax.runtime.notifications.models import NotificationMessage

logger = logging.getLogger(__name__)


class LoggingNotificationAdapter:
    """Default Tier-0 adapter — logs payloads without network calls."""

    async def notify(self, message: NotificationMessage) -> None:
        logger.info(
            "notification channel=%s task=%s subject=%s body=%s metadata=%s",
            message.channel,
            message.task_id,
            message.subject,
            message.body,
            message.metadata,
        )
