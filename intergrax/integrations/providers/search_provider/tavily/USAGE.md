# `tavily` integration — usage

**Category:** ``search_provider``  
**Catalog factory:** ``create_tavily_search_provider()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(search_provider="tavily")
backend = profile.resolve(IntegrationCategory.SEARCH_PROVIDER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider

backend = create_tavily_search_provider(**config_overrides)
```


## Environment variables

`INTERGRAX_TAVILY_API_KEY`, optional `INTERGRAX_TAVILY_URL`

## Example

```python
from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider

search = create_tavily_search_provider(api_key="tvly-...")
hits = search.search("agent harness", limit=5)
```

## Notes

Agent-native research API (Phase M.7). Thin shell → ``_shared/p3/factories``.
