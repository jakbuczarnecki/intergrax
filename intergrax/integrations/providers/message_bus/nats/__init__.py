# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_nats_message_bus", "register_nats_integration"]

def __getattr__(name: str):
    if name == "register_nats_integration":
        from intergrax.integrations.providers.message_bus.nats.register import register_nats_integration
        return register_nats_integration
    if name == "create_nats_message_bus":
        from intergrax.integrations.providers.message_bus.nats.bundle import create_nats_message_bus
        return create_nats_message_bus
    raise AttributeError(name)
