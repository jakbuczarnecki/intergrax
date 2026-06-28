# Stripe (stripe)

Category: `billing_meter`

## Legacy facade

- `create_stripe_billing_meter()` remains backward-compatible.

## Contract-based integration

- `StripeBillingMeterIntegration` derives from the category-specific contract.
- Factory: `create_stripe_billing_meter_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
