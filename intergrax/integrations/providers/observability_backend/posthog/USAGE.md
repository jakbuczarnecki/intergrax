# Posthog (posthog)

Category: `observability_backend`

## Single public entrypoint

- **`PosthogObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PosthogObservabilityIntegration`.
- Contract factory: `create_posthog_observability_backend_integration()`.
