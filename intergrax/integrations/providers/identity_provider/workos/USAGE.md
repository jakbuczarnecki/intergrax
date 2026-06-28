# Workos (workos)

Category: `identity_provider`

## Legacy facade

- `create_workos_identity_provider()` remains backward-compatible.

## Contract-based integration

- `WorkosIdentityProviderIntegration` derives from the category-specific contract.
- Factory: `create_workos_identity_provider_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
