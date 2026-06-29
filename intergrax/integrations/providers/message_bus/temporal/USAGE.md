# Temporal (temporal)

Category: `message_bus`

## Single public entrypoint

- **`TemporalMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TemporalMessageBusIntegration`.
- Contract factory: `create_temporal_message_bus_integration()`.
