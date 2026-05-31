# `confluence` integration — usage

**Category:** ``wiki_knowledge``  
**Catalog factory:** ``create_confluence_wiki_knowledge()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(wiki_knowledge=IntegrationSlug.CONFLUENCE)
backend = profile.resolve(IntegrationCategory.WIKI_KNOWLEDGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import create_confluence_wiki_knowledge

backend = create_confluence_wiki_knowledge(**config_overrides)
```


## Environment variables

`INTERGRAX_CONFLUENCE_BASE_URL`, `INTERGRAX_CONFLUENCE_EMAIL`, `INTERGRAX_CONFLUENCE_API_TOKEN`

## Example

```python
from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import create_confluence_wiki_knowledge

wiki = create_confluence_wiki_knowledge(base_url="https://acme.atlassian.net/wiki", email="...", api_token="...")
page = wiki.get_page("123456")
results = wiki.search_pages("runbook deployment", limit=10)
```

## Notes

httpx only in ``opens.py``.
