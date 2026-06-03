# `sharepoint` integration — usage

**Category:** ``wiki_knowledge``  
**Catalog factory:** ``create_sharepoint_wiki_knowledge()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(wiki_knowledge="sharepoint")
backend = profile.resolve(IntegrationCategory.WIKI_KNOWLEDGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import create_sharepoint_wiki_knowledge

backend = create_sharepoint_wiki_knowledge(**config_overrides)
```


## Environment variables

`INTERGRAX_SHAREPOINT_TOKEN`; optional `INTERGRAX_SHAREPOINT_SITE_URL`, `INTERGRAX_SHAREPOINT_URL`

## Example

```python
from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import create_sharepoint_wiki_knowledge

wiki = create_sharepoint_wiki_knowledge(token="...", site_url="https://contoso.sharepoint.com/sites/docs")
page = wiki.get_page("page-id")
results = wiki.search_pages("incident response", limit=10)
```

## Notes

Microsoft Graph / SharePoint REST via httpx.
