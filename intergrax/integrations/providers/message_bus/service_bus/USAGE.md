# Service Bus (service_bus)

Category: `message_bus`

## Single public entrypoint

- **`ServiceBusMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ServiceBusMessageBusIntegration`.
- Contract factory: `create_service_bus_message_bus_integration()`.
