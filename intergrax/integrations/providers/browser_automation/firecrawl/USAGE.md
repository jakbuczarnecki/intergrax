# `firecrawl` integration — usage

**Category:** ``browser_automation``  
**Catalog factory:** ``create_firecrawl_browser_automation()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(browser_automation="firecrawl")
backend = profile.resolve(IntegrationCategory.BROWSER_AUTOMATION)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.browser_automation.firecrawl.bundle import create_firecrawl_browser_automation

backend = create_firecrawl_browser_automation(**config_overrides)
```


## Environment variables

`INTERGRAX_FIRECRAWL_API_KEY`

## Example

```python
from intergrax.integrations.providers.browser_automation.firecrawl.bundle import create_firecrawl_browser_automation

crawl = create_firecrawl_browser_automation(api_key="fc-...")
page = crawl.fetch_page("https://docs.example.com")
```

## Notes

Structured crawl API — alternative to raw Playwright.
