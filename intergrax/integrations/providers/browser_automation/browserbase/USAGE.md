# Browserbase (browserbase)

Category: `browser_automation`

## Legacy facade

- `create_browserbase_browser_automation()` remains backward-compatible.

## Contract-based integration

- `BrowserbaseBrowserAutomationIntegration` derives from the category-specific contract.
- Factory: `create_browserbase_browser_automation_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
