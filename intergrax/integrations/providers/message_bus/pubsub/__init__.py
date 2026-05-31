# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_pubsub_message_bus", "register_pubsub_integration"]

def __getattr__(name: str):
    if name == "register_pubsub_integration":
        from intergrax.integrations.providers.message_bus.pubsub.register import register_pubsub_integration
        return register_pubsub_integration
    if name == "create_pubsub_message_bus":
        from intergrax.integrations.providers.message_bus.pubsub.bundle import create_pubsub_message_bus
        return create_pubsub_message_bus
    raise AttributeError(name)
