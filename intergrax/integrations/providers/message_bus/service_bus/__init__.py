# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_service_bus_message_bus", "register_service_bus_integration"]

def __getattr__(name: str):
    if name == "register_service_bus_integration":
        from intergrax.integrations.providers.message_bus.service_bus.register import register_service_bus_integration
        return register_service_bus_integration
    if name == "create_service_bus_message_bus":
        from intergrax.integrations.providers.message_bus.service_bus.bundle import create_service_bus_message_bus
        return create_service_bus_message_bus
    raise AttributeError(name)
