# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stripe billing meter integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.billing_meter import BillingMeterBackend
from intergrax.runtime.integrations.categories.automation import BillingMeterIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

STRIPE_BILLING_METER_PROVIDER_ID = "stripe"


class StripeBillingMeterIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Stripe billing meter integration."""

    pass


StripeBillingMeterClient = BillingMeterBackend

class StripeBillingMeterIntegration(BillingMeterIntegrationContract):
    """
    Single public Stripe billing meter entrypoint.

    Legacy catalog factory (create_stripe_billing_meter) owns catalog behavior; legacy factories use from_client().
    """

    config: StripeBillingMeterIntegrationConfig = StripeBillingMeterIntegrationConfig()
    _client: StripeBillingMeterClient | None = PrivateAttr(default=None)
    

    def list_meter_events(self, customer_id, limit: int = 50):
        return self._require_client().list_meter_events(customer_id, limit=limit)

    def submit_meter_event(self, customer_id, metric, quantity):
        return self._require_client().submit_meter_event(customer_id, metric, quantity)

    def _require_client(self) -> BillingMeterBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: StripeBillingMeterClient,
        *,
        enabled: bool = False,
    ) -> StripeBillingMeterIntegration:
        integration = cls.for_provider(
            provider_id=STRIPE_BILLING_METER_PROVIDER_ID,
            display_name="Stripe",
            config=StripeBillingMeterIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> StripeBillingMeterClient | None:
        return self._client

BillingMeterBackend.register(StripeBillingMeterIntegration)
