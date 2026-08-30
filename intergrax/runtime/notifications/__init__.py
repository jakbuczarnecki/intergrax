# © Artur Czarnecki. All rights reserved.

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
from intergrax.runtime.notifications.delivery_contract import (
    NotificationDelivery,
    NullNotificationDelivery,
)
from intergrax.runtime.notifications.backend_contract import (
    NotificationBackend,
    NotificationSettings,
    resolve_notification_settings,
)
from intergrax.runtime.notifications.factory import (
    create_notification_adapter,
    resolve_notification_adapter,
)
from intergrax.runtime.notifications.models import NotificationMessage

__all__ = [
    "LoggingNotificationAdapter",
    "NotificationAdapter",
    "NotificationBackend",
    "NotificationDelivery",
    "NotificationMessage",
    "NotificationSettings",
    "NullNotificationDelivery",
    "WebhookNotificationAdapter",
    "create_notification_adapter",
    "resolve_notification_adapter",
    "resolve_notification_settings",
]
