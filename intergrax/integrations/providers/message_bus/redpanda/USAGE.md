# Redpanda (redpanda)

Category: `message_bus`

## Single public entrypoint

- **`RedpandaMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `RedpandaMessageBusIntegration`.
- Contract factory: `create_redpanda_message_bus_integration()`.
