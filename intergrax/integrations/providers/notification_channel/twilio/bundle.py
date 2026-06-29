# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_twilio_notification_channel as _legacy_create_twilio_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.twilio.integration import (
    TWILIO_NOTIFICATION_CHANNEL_PROVIDER_ID,
    TwilioNotificationChannelIntegration,
    TwilioNotificationChannelIntegrationConfig,
    TwilioNotificationChannelClient,
)

__all__ = [
    "create_twilio_notification_channel",
    "create_twilio_notification_channel_integration",
]


def create_twilio_notification_channel_integration(
    *,
    client: TwilioNotificationChannelClient | None = None,
    enabled: bool = False,
) -> TwilioNotificationChannelIntegration:
    """
    Build a contract-based Twilio notification channel integration.

    The legacy facade (create_twilio_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Twilio notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TwilioNotificationChannelIntegration.from_client(client, enabled=enabled)
    return TwilioNotificationChannelIntegration.for_provider(
        provider_id=TWILIO_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Twilio",
        config=TwilioNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_twilio_notification_channel(**kwargs: object) -> TwilioNotificationChannelIntegration:
    """Compatibility shim — constructs TwilioNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_twilio_notification_channel(**kwargs)
    if isinstance(runtime, TwilioNotificationChannelIntegration):
        return runtime
    return TwilioNotificationChannelIntegration.from_client(runtime)
