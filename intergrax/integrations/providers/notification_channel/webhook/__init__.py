# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Generic webhook integration — single public entry for HTTP outbound notifications.

Implementation lives under ``runtime.notifications.adapters.webhook_adapter``;
compose only through this package.
"""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.notification_channel.webhook.config import (
    DEFAULT_WEBHOOK_URL,
    ENV_WEBHOOK_URL,
    WebhookIntegrationConfig,
)

__all__ = [
    "DEFAULT_WEBHOOK_URL",
    "ENV_WEBHOOK_URL",
    "WebhookIntegrationBundle",
    "WebhookIntegrationConfig",
    "create_webhook_integration",
    "create_webhook_notification_channel",
    "register_webhook_integration",
    "resolve_webhook_config",
    "create_webhook_notification_channel_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "WebhookIntegrationBundle",
        "create_webhook_integration",
        "create_webhook_notification_channel",
        "resolve_webhook_config",
        "create_webhook_notification_channel_integration",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "WEBHOOK_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "WebhookNotificationChannelIntegration",
        "WebhookNotificationChannelIntegrationConfig",
        "WebhookNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_webhook_integration":
        from intergrax.integrations.providers.notification_channel.webhook.register import register_webhook_integration

        return register_webhook_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.webhook import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.webhook import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
