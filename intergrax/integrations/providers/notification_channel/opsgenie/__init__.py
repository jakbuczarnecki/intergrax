# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_opsgenie_notification_channel", "register_opsgenie_integration"]

def __getattr__(name: str):
    if name == "register_opsgenie_integration":
        from intergrax.integrations.providers.notification_channel.opsgenie.register import register_opsgenie_integration
        return register_opsgenie_integration
    if name == "create_opsgenie_notification_channel":
        from intergrax.integrations.providers.notification_channel.opsgenie.bundle import create_opsgenie_notification_channel
        return create_opsgenie_notification_channel
    raise AttributeError(name)
