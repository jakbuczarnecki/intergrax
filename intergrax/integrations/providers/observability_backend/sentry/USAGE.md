# Sentry (sentry)

Category: `observability_backend`

## Single public entrypoint

- **`SentryObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SentryObservabilityIntegration`.
- Contract factory: `create_sentry_observability_backend_integration()`.
