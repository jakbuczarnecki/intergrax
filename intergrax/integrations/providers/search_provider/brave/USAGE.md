# `brave` integration — usage

**Category:** ``search_provider``  
**Catalog factory:** ``create_brave_search_provider()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(search_provider="brave")
backend = profile.resolve(IntegrationCategory.SEARCH_PROVIDER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.search_provider.brave.bundle import create_brave_search_provider

backend = create_brave_search_provider(**config_overrides)
```


## Environment variables

`INTERGRAX_BRAVE_API_KEY`

## Example

```python
from intergrax.integrations.providers.search_provider.brave.bundle import create_brave_search_provider

search = create_brave_search_provider(api_key="BSA...")
hits = search.search("Intergrax agent orchestration", limit=5)
for hit in hits:
    print(hit.rank, hit.title, hit.url)
```

## Notes

Brave Web Search API via httpx. Hit normalization in ``_shared/rest_search.py``.
