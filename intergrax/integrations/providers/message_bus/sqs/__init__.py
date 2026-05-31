# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_sqs_message_bus", "register_sqs_integration"]

def __getattr__(name: str):
    if name == "register_sqs_integration":
        from intergrax.integrations.providers.message_bus.sqs.register import register_sqs_integration
        return register_sqs_integration
    if name == "create_sqs_message_bus":
        from intergrax.integrations.providers.message_bus.sqs.bundle import create_sqs_message_bus
        return create_sqs_message_bus
    raise AttributeError(name)
