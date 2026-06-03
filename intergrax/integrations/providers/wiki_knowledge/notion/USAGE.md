# `notion` integration — usage

**Category:** ``wiki_knowledge``  
**Catalog factory:** ``create_notion_wiki_knowledge()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(wiki_knowledge="notion")
backend = profile.resolve(IntegrationCategory.WIKI_KNOWLEDGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.wiki_knowledge.notion.bundle import create_notion_wiki_knowledge

backend = create_notion_wiki_knowledge(**config_overrides)
```


## Environment variables

`INTERGRAX_NOTION_API_KEY` (Bearer token); optional `INTERGRAX_NOTION_URL`

## Example

```python
from intergrax.integrations.providers.wiki_knowledge.notion.bundle import create_notion_wiki_knowledge

wiki = create_notion_wiki_knowledge(api_key="secret_...")
page = wiki.get_page("page-uuid")
results = wiki.search_pages("deployment runbook", limit=10)
```

## Notes

Notion REST API via httpx. Complements ``confluence`` for mixed knowledge bases.
