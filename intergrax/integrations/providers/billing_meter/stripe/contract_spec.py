# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Stripe billing meter."""

from __future__ import annotations

from intergrax.integrations.providers.billing_meter.stripe.bundle import (
    create_stripe_billing_meter_integration,
)
from intergrax.integrations.providers.billing_meter.stripe.integration import (
    STRIPE_BILLING_METER_PROVIDER_ID,
    StripeBillingMeterIntegration,
    StripeBillingMeterIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.automation import BillingMeterIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="billing_meter",
    provider_id=STRIPE_BILLING_METER_PROVIDER_ID,
    integration_class=StripeBillingMeterIntegration,
    contract_class=BillingMeterIntegrationContract,
    contract_factory=create_stripe_billing_meter_integration,
    display_name="Stripe",
    config_class=StripeBillingMeterIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
