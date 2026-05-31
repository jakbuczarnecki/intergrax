# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_temporal_message_bus", "register_temporal_integration"]

def __getattr__(name: str):
    if name == "register_temporal_integration":
        from intergrax.integrations.providers.message_bus.temporal.register import register_temporal_integration
        return register_temporal_integration
    if name == "create_temporal_message_bus":
        from intergrax.integrations.providers.message_bus.temporal.bundle import create_temporal_message_bus
        return create_temporal_message_bus
    raise AttributeError(name)
