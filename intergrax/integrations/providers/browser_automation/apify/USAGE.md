# Apify (apify)

Category: `browser_automation`

## Single public entrypoint

- **`ApifyBrowserAutomationIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ApifyBrowserAutomationIntegration`.
- Contract factory: `create_apify_browser_automation_integration()`.
