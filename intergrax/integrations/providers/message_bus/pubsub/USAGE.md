# Pubsub (pubsub)

Category: `message_bus`

## Single public entrypoint

- **`PubsubMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PubsubMessageBusIntegration`.
- Contract factory: `create_pubsub_message_bus_integration()`.
