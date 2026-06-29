# Confluent (confluent)

Category: `message_bus`

## Single public entrypoint

- **`ConfluentMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ConfluentMessageBusIntegration`.
- Contract factory: `create_confluent_message_bus_integration()`.
