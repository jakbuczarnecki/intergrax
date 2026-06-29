# Nats (nats)

Category: `message_bus`

## Single public entrypoint

- **`NatsMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `NatsMessageBusIntegration`.
- Contract factory: `create_nats_message_bus_integration()`.
