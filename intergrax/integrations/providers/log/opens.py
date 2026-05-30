# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal openers for log notification integration."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.log.adapter import LogNotificationAdapter
from intergrax.integrations.providers.log.config import LogIntegrationConfig
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter


def open_log_notification_channel(
    config: LogIntegrationConfig,
    *,
    implementation: Optional[NotificationAdapter] = None,
) -> NotificationAdapter:
    _ = config
    if implementation is not None:
        return implementation
    return LogNotificationAdapter()
