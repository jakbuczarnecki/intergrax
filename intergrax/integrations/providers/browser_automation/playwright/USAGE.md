# `playwright` integration — usage

**Category:** ``browser_automation``  
**Catalog factory:** ``create_playwright_browser_automation()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(browser_automation=IntegrationSlug.PLAYWRIGHT)
backend = profile.resolve(IntegrationCategory.BROWSER_AUTOMATION)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.browser_automation.playwright.bundle import create_playwright_browser_automation

backend = create_playwright_browser_automation(**config_overrides)
```


## Environment variables

Optional overrides: ``headless=True``, ``timeout_ms=30000`` (no required env vars)

## Example

```python
from intergrax.integrations.providers.browser_automation.playwright.bundle import create_playwright_browser_automation

browser = create_playwright_browser_automation(headless=True, timeout_ms=30000)
page = browser.fetch_page("https://example.com/dashboard", wait_until="networkidle")
print(page.title, page.text[:200])
browser.close()
```

## Notes

``playwright`` Chromium launch opened lazily. Use for JS-heavy pages; prefer ``search_provider`` for simple research.
