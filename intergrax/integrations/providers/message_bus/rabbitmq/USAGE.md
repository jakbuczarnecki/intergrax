# Rabbitmq (rabbitmq)

Category: `message_bus`

## Single public entrypoint

- **`RabbitmqMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `RabbitmqMessageBusIntegration`.
- Contract factory: `create_rabbitmq_message_bus_integration()`.
