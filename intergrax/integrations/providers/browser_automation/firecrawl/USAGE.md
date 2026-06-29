# Firecrawl (firecrawl)

Category: `browser_automation`

## Single public entrypoint

- **`FirecrawlBrowserAutomationIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `FirecrawlBrowserAutomationIntegration`.
- Contract factory: `create_firecrawl_browser_automation_integration()`.
