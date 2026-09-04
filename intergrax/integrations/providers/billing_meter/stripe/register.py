# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register stripe in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.billing_meter.stripe.bundle import create_stripe_billing_meter
from intergrax.integrations.providers.billing_meter.stripe.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.billing_meter.stripe.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_stripe_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_stripe_billing_meter,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
