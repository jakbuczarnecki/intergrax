# Mailgun (mailgun)

Category: `interaction_surface`

## Legacy facade

- `create_mailgun_interaction_surface()` remains backward-compatible.

## Contract-based integration

- `MailgunInteractionSurfaceIntegration` derives from the category-specific contract.
- Factory: `create_mailgun_interaction_surface_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
