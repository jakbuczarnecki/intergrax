# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_telegram_catalog_factory as _legacy_create_telegram_catalog_factory

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.telegram.integration import (
    TELEGRAM_NOTIFICATION_CHANNEL_PROVIDER_ID,
    TelegramNotificationChannelIntegration,
    TelegramNotificationChannelIntegrationConfig,
    TelegramNotificationChannelClient,
)

__all__ = [
    "create_telegram_catalog_factory",
    "create_telegram_notification_channel_integration",
]


def create_telegram_notification_channel_integration(
    *,
    client: TelegramNotificationChannelClient | None = None,
    enabled: bool = False,
) -> TelegramNotificationChannelIntegration:
    """
    Build a contract-based Telegram notification channel integration.

    The legacy facade (create_telegram_catalog_factory) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Telegram notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TelegramNotificationChannelIntegration.from_client(client, enabled=enabled)
    return TelegramNotificationChannelIntegration.for_provider(
        provider_id=TELEGRAM_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Telegram",
        config=TelegramNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_telegram_catalog_factory(**kwargs: object) -> TelegramNotificationChannelIntegration:
    """Compatibility shim — constructs TelegramNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_telegram_catalog_factory(**kwargs)
    if isinstance(runtime, TelegramNotificationChannelIntegration):
        return runtime
    return TelegramNotificationChannelIntegration.from_runtime(runtime)
