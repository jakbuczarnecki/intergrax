# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_incident_io_notification_channel", "register_incident_io_integration"]

def __getattr__(name: str):
    if name == "register_incident_io_integration":
        from intergrax.integrations.providers.notification_channel.incident_io.register import register_incident_io_integration
        return register_incident_io_integration
    if name == "create_incident_io_notification_channel":
        from intergrax.integrations.providers.notification_channel.incident_io.bundle import create_incident_io_notification_channel
        return create_incident_io_notification_channel
    raise AttributeError(name)
