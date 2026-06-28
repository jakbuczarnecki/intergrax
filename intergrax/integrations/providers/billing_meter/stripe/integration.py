# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stripe billing meter integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import BillingMeterIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

STRIPE_BILLING_METER_PROVIDER_ID = "stripe"


class StripeBillingMeterIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Stripe billing meter integration."""

    pass


@runtime_checkable
class StripeBillingMeterClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class StripeBillingMeterIntegration(BillingMeterIntegrationContract):
    """
    Stripe billing meter integration.

    The legacy facade (create_stripe_billing_meter) remains separate and backward-compatible.
    """

    config: StripeBillingMeterIntegrationConfig = StripeBillingMeterIntegrationConfig()
    _client: StripeBillingMeterClient | None = PrivateAttr(default=None)

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
