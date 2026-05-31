# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_pagerduty_notification_channel", "register_pagerduty_integration"]

def __getattr__(name: str):
    if name == "register_pagerduty_integration":
        from intergrax.integrations.providers.notification_channel.pagerduty.register import register_pagerduty_integration
        return register_pagerduty_integration
    if name == "create_pagerduty_notification_channel":
        from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel
        return create_pagerduty_notification_channel
    raise AttributeError(name)
