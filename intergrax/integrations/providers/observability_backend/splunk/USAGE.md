# Splunk (splunk)

Category: `observability_backend`

## Single public entrypoint

- **`SplunkObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SplunkObservabilityIntegration`.
- Contract factory: `create_splunk_observability_backend_integration()`.
