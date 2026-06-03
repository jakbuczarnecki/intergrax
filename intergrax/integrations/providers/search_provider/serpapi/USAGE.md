# `serpapi` integration — usage

**Category:** ``search_provider``  
**Catalog factory:** ``create_serpapi_search_provider()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(search_provider="serpapi")
backend = profile.resolve(IntegrationCategory.SEARCH_PROVIDER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.search_provider.serpapi.bundle import create_serpapi_search_provider

backend = create_serpapi_search_provider(**config_overrides)
```


## Environment variables

`INTERGRAX_SERPAPI_API_KEY`

## Example

```python
from intergrax.integrations.providers.search_provider.serpapi.bundle import create_serpapi_search_provider

search = create_serpapi_search_provider(api_key="...")
hits = search.search("enterprise AI agents", limit=5)
```

## Notes

SerpAPI JSON API via httpx.
