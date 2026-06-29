# Playwright (playwright)

Category: `browser_automation`

## Single public entrypoint

- **`PlaywrightBrowserAutomationIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PlaywrightBrowserAutomationIntegration`.
- Contract factory: `create_playwright_browser_automation_integration()`.
