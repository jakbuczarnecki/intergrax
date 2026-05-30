# `bing` integration — usage

**Category:** ``search_provider``  
**Catalog factory:** ``create_bing_search_provider()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(search_provider=IntegrationSlug.BING)
backend = profile.resolve(IntegrationCategory.SEARCH_PROVIDER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.bing.bundle import create_bing_search_provider

backend = create_bing_search_provider(**config_overrides)
```


## Environment variables

`INTERGRAX_BING_API_KEY` (legacy: `BING_SEARCH_V7_API_KEY`)

## Example

```python
from intergrax.integrations.providers.bing.bundle import create_bing_search_provider

search = create_bing_search_provider(api_key="...")
hits = search.search("enterprise AI agents", limit=5)
```

## Notes

HTTP client only in ``opens.py``.
