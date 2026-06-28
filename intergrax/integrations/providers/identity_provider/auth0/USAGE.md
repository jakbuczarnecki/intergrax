# Auth0 (auth0)

Category: `identity_provider`

## Legacy facade

- `create_auth0_identity_provider()` remains backward-compatible.

## Contract-based integration

- `Auth0IdentityProviderIntegration` derives from the category-specific contract.
- Factory: `create_auth0_identity_provider_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
