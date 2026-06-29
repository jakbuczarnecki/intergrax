# Elasticsearch (elasticsearch)

Category: `observability_backend`

## Single public entrypoint

- **`ElasticsearchObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ElasticsearchObservabilityIntegration`.
- Contract factory: `create_elasticsearch_observability_backend_integration()`.
