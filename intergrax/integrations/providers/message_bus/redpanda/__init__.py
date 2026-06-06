# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_redpanda_message_bus", "register_redpanda_integration"]

def __getattr__(name: str):
    if name == "register_redpanda_integration":
        from intergrax.integrations.providers.message_bus.redpanda.register import register_redpanda_integration
        return register_redpanda_integration
    if name == "create_redpanda_message_bus":
        from intergrax.integrations.providers.message_bus.redpanda.bundle import create_redpanda_message_bus
        return create_redpanda_message_bus
    raise AttributeError(name)
