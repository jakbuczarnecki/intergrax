# Browserbase (browserbase)

Category: `browser_automation`

## Single public entrypoint

- **`BrowserbaseBrowserAutomationIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BrowserbaseBrowserAutomationIntegration`.
- Contract factory: `create_browserbase_browser_automation_integration()`.
