# Celery (celery)

Category: `message_bus`

## Single public entrypoint

- **`CeleryMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CeleryMessageBusIntegration`.
- Contract factory: `create_celery_message_bus_integration()`.
