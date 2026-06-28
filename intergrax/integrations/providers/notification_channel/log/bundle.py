# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Log notification integration bundle (Phase M.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.providers.notification_channel.log.config import LogIntegrationConfig
from intergrax.integrations.providers.notification_channel.log.opens import open_log_notification_channel
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter


@dataclass(frozen=True)
class LogIntegrationBundle:
    config: LogIntegrationConfig
    notification_channel: NotificationAdapter


def resolve_log_config(**overrides: object) -> LogIntegrationConfig:
    return LogIntegrationConfig.from_env(**overrides)


def create_log_integration(
    *,
    notification_adapter: Optional[NotificationAdapter] = None,
    **config_overrides: object,
) -> LogIntegrationBundle:
    config = resolve_log_config(**config_overrides)
    channel = open_log_notification_channel(config, implementation=notification_adapter)
    return LogIntegrationBundle(config=config, notification_channel=channel)


def create_log_notification_channel(
    *,
    notification_adapter: Optional[NotificationAdapter] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Catalog factory for ``"log"`` / ``NOTIFICATION_CHANNEL``."""
    return create_log_integration(
        notification_adapter=notification_adapter,
        **config_overrides,
    ).notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.log.integration import (
    LOG_NOTIFICATION_CHANNEL_PROVIDER_ID,
    LogNotificationChannelIntegration,
    LogNotificationChannelIntegrationConfig,
    LogNotificationChannelClient,
)


def create_log_notification_channel_integration(
    *,
    client: LogNotificationChannelClient | None = None,
    enabled: bool = False,
) -> LogNotificationChannelIntegration:
    """
    Build a contract-based Log notification channel integration.

    The legacy facade (create_log_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Log notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LogNotificationChannelIntegration.from_client(client, enabled=enabled)
    return LogNotificationChannelIntegration.for_provider(
        provider_id=LOG_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Log",
        config=LogNotificationChannelIntegrationConfig(enabled=enabled),
    )
