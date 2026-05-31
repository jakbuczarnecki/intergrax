# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_twilio_notification_channel", "register_twilio_integration"]

def __getattr__(name: str):
    if name == "register_twilio_integration":
        from intergrax.integrations.providers.notification_channel.twilio.register import register_twilio_integration
        return register_twilio_integration
    if name == "create_twilio_notification_channel":
        from intergrax.integrations.providers.notification_channel.twilio.bundle import create_twilio_notification_channel
        return create_twilio_notification_channel
    raise AttributeError(name)
