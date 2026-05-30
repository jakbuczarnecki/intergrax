# `elasticsearch` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_elasticsearch_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(observability_backend=IntegrationSlug.ELASTICSEARCH)
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.elasticsearch.bundle import create_elasticsearch_observability_backend

backend = create_elasticsearch_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_ELASTICSEARCH_URL`, `INTERGRAX_ELASTICSEARCH_INDEX`; optional USER/PASSWORD/API_KEY

## Example

```python
from intergrax.integrations.providers.elasticsearch.bundle import create_elasticsearch_observability_backend

obs = create_elasticsearch_observability_backend(base_url="http://localhost:9200", index="logs-*")
# The promql argument is a Lucene query_string (not PromQL):
instant = obs.query_instant('level:"error" AND service:"nexus"')
range_result = obs.query_range('status:500', start=1710000000, end=1710003600, step="15s")
```

## Notes

``promql`` in the API maps to Lucene ``query_string``. httpx only in ``opens.py``.
