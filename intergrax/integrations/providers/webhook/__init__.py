# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Generic webhook integration — single public entry for HTTP outbound notifications.

Implementation lives under ``runtime.notifications.adapters.webhook_adapter``;
compose only through this package.
"""

from intergrax.integrations.providers.webhook.config import (
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
]

_BUNDLE_EXPORTS = frozenset(
    {
        "WebhookIntegrationBundle",
        "create_webhook_integration",
        "create_webhook_notification_channel",
        "resolve_webhook_config",
    }
)


def __getattr__(name: str):
    if name == "register_webhook_integration":
        from intergrax.integrations.providers.webhook.register import register_webhook_integration

        return register_webhook_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.webhook import bundle as _bundle

        return getattr(_bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
