# `selenium` integration — usage

**Category:** ``browser_automation``  
**Catalog factory:** ``create_selenium_browser_automation()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(browser_automation="selenium")
backend = profile.resolve(IntegrationCategory.BROWSER_AUTOMATION)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.browser_automation.selenium.bundle import create_selenium_browser_automation

backend = create_selenium_browser_automation(**config_overrides)
```


## Environment variables

`INTERGRAX_SELENIUM_DRIVER_URL` (optional remote grid), `INTERGRAX_SELENIUM_BROWSER`

## Example

```python
from intergrax.integrations.providers.browser_automation.selenium.bundle import create_selenium_browser_automation

browser = create_selenium_browser_automation(headless=True)
page = browser.fetch_page("https://legacy.example.com")
```

## Notes

Legacy browser stacks; requires ``selenium`` package.
