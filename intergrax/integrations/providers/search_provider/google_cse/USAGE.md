# Google Cse (google_cse)

Category: `search_provider`

## Legacy facade

- `create_google_cse_integration()` remains backward-compatible.

## Contract-based integration

- `GoogleCseSearchProviderIntegration` derives from the category-specific contract.
- Factory: `create_google_cse_search_provider_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
