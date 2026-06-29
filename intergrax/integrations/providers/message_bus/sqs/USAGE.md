# Sqs (sqs)

Category: `message_bus`

## Single public entrypoint

- **`SqsMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SqsMessageBusIntegration`.
- Contract factory: `create_sqs_message_bus_integration()`.
