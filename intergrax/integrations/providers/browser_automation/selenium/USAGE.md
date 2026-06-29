# Selenium (selenium)

Category: `browser_automation`

## Single public entrypoint

- **`SeleniumBrowserAutomationIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SeleniumBrowserAutomationIntegration`.
- Contract factory: `create_selenium_browser_automation_integration()`.
