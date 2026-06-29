# Influxdb (influxdb)

Category: `observability_backend`

## Single public entrypoint

- **`InfluxdbObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `InfluxdbObservabilityIntegration`.
- Contract factory: `create_influxdb_observability_backend_integration()`.
