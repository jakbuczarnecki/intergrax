# `google_cse` integration — usage

**Category:** ``search_provider``  
**Catalog factory:** ``create_google_cse_search_provider()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(search_provider=IntegrationSlug.GOOGLE_CSE)
backend = profile.resolve(IntegrationCategory.SEARCH_PROVIDER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.google_cse.bundle import create_google_cse_search_provider

backend = create_google_cse_search_provider(**config_overrides)
```


## Environment variables

`INTERGRAX_GOOGLE_CSE_API_KEY`, `INTERGRAX_GOOGLE_CSE_CX`

## Example

```python
from intergrax.integrations.providers.google_cse.bundle import create_google_cse_search_provider

search = create_google_cse_search_provider(api_key="...", cx="...")
hits = search.search("Intergrax agent orchestration", limit=5)
for hit in hits:
    print(hit.title, hit.url)
```

## Notes

Compatible with ``WebSearchExecutor`` via ``search.web_search_provider``.
