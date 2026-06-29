# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stripe billing meter integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.billing_meter import BillingMeterBackend
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
    Single public Stripe billing meter entrypoint.

    Legacy catalog factory (create_stripe_billing_meter) delegates to this class.
    """

    config: StripeBillingMeterIntegrationConfig = StripeBillingMeterIntegrationConfig()
    _client: StripeBillingMeterClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> StripeBillingMeterIntegration:
        integration = cls.for_provider(
            provider_id=STRIPE_BILLING_METER_PROVIDER_ID,
            display_name="Stripe",
            config=StripeBillingMeterIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Stripe integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

BillingMeterBackend.register(StripeBillingMeterIntegration)
