# Pulsar (pulsar)

Category: `message_bus`

## Single public entrypoint

- **`PulsarMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PulsarMessageBusIntegration`.
- Contract factory: `create_pulsar_message_bus_integration()`.
