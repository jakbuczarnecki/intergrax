# Stripe (stripe)

Category: `billing_meter`

## Single public entrypoint

- **`StripeBillingMeterIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `StripeBillingMeterIntegration`.
- Contract factory: `create_stripe_billing_meter_integration()`.
