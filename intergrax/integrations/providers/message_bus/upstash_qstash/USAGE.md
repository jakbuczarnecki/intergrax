# Upstash Qstash (upstash_qstash)

Category: `message_bus`

## Single public entrypoint

- **`UpstashQstashMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `UpstashQstashMessageBusIntegration`.
- Contract factory: `create_upstash_qstash_message_bus_integration()`.
