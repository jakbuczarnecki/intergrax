# Clickhouse (clickhouse)

Category: `observability_backend`

## Single public entrypoint

- **`ClickhouseObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ClickhouseObservabilityIntegration`.
- Contract factory: `create_clickhouse_observability_backend_integration()`.
