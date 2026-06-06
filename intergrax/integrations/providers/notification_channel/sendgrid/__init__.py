# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_sendgrid_notification_channel", "register_sendgrid_integration"]

def __getattr__(name: str):
    if name == "register_sendgrid_integration":
        from intergrax.integrations.providers.notification_channel.sendgrid.register import register_sendgrid_integration
        return register_sendgrid_integration
    if name == "create_sendgrid_notification_channel":
        from intergrax.integrations.providers.notification_channel.sendgrid.bundle import create_sendgrid_notification_channel
        return create_sendgrid_notification_channel
    raise AttributeError(name)
