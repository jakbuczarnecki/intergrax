# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal openers for log notification integration."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.notification_channel.log.adapter import _LogNotificationAdapter
from intergrax.integrations.providers.notification_channel.log.config import LogIntegrationConfig
from intergrax.integrations.providers.notification_channel.log.integration import LogNotificationChannelIntegration
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter


def open_log_notification_channel(
    config: LogIntegrationConfig,
    *,
    implementation: Optional[NotificationAdapter] = None,
) -> LogNotificationChannelIntegration:
    _ = config
    if implementation is not None:
        if isinstance(implementation, LogNotificationChannelIntegration):
            return implementation
        return LogNotificationChannelIntegration.from_client(implementation)
    return LogNotificationChannelIntegration.from_client(_LogNotificationAdapter())
