# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_stripe_billing_meter as _legacy_create_stripe_billing_meter

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.billing_meter.stripe.integration import (
    STRIPE_BILLING_METER_PROVIDER_ID,
    StripeBillingMeterIntegration,
    StripeBillingMeterIntegrationConfig,
    StripeBillingMeterClient,
)

__all__ = [
    "create_stripe_billing_meter",
    "create_stripe_billing_meter_integration",
]


def create_stripe_billing_meter_integration(
    *,
    client: StripeBillingMeterClient | None = None,
    enabled: bool = False,
) -> StripeBillingMeterIntegration:
    """
    Build a contract-based Stripe billing meter integration.

    The legacy facade (create_stripe_billing_meter) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Stripe billing meter integration requires an injected client when enabled=True",
        )
    if client is not None:
        return StripeBillingMeterIntegration.from_client(client, enabled=enabled)
    return StripeBillingMeterIntegration.for_provider(
        provider_id=STRIPE_BILLING_METER_PROVIDER_ID,
        display_name="Stripe",
        config=StripeBillingMeterIntegrationConfig(enabled=enabled),
    )


def create_stripe_billing_meter(**kwargs: object) -> StripeBillingMeterIntegration:
    """Compatibility shim — constructs StripeBillingMeterIntegration from legacy runtime."""
    runtime = _legacy_create_stripe_billing_meter(**kwargs)
    if isinstance(runtime, StripeBillingMeterIntegration):
        return runtime
    return StripeBillingMeterIntegration.from_client(runtime)
