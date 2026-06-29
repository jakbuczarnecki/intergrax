# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "STRIPE_BILLING_METER_PROVIDER_ID",
    "StripeBillingMeterIntegration",
    "StripeBillingMeterIntegrationConfig",
    "StripeBillingMeterClient",
    "create_stripe_billing_meter",
    "create_stripe_billing_meter_integration",
    "register_stripe_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_stripe_billing_meter",
        "create_stripe_billing_meter_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "STRIPE_BILLING_METER_PROVIDER_ID",
        "StripeBillingMeterIntegration",
        "StripeBillingMeterIntegrationConfig",
        "StripeBillingMeterClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "STRIPE_BILLING_METER_PROVIDER_ID",
        "StripeBillingMeterIntegration",
        "StripeBillingMeterIntegrationConfig",
        "StripeBillingMeterClient",
    }
)

def __getattr__(name: str):
    if name == "register_stripe_integration":
        from intergrax.integrations.providers.billing_meter.stripe.register import register_stripe_integration

        return register_stripe_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.billing_meter.stripe import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.billing_meter.stripe import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.billing_meter.stripe import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
