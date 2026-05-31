# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_email_smtp_notification_channel", "register_email_smtp_integration"]

def __getattr__(name: str):
    if name == "register_email_smtp_integration":
        from intergrax.integrations.providers.notification_channel.email_smtp.register import register_email_smtp_integration
        return register_email_smtp_integration
    if name == "create_email_smtp_notification_channel":
        from intergrax.integrations.providers.notification_channel.email_smtp.bundle import create_email_smtp_notification_channel
        return create_email_smtp_notification_channel
    raise AttributeError(name)
