# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Opsgenie notification channel integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID = "opsgenie"


class OpsgenieNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Opsgenie notification channel integration."""

    pass


@runtime_checkable
class OpsgenieNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class OpsgenieNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Single public Opsgenie notification channel entrypoint.

    Legacy catalog factory (create_opsgenie_notification_channel) delegates to this class.
    """

    config: OpsgenieNotificationChannelIntegrationConfig = OpsgenieNotificationChannelIntegrationConfig()
    _client: OpsgenieNotificationChannelClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> OpsgenieNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Opsgenie",
            config=OpsgenieNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    async def notify(self, message: Any) -> None:
        await self._require_runtime().notify(message)

    def health(self) -> Any:
        return self._require_runtime().health()


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


    @classmethod
    def from_client(
        cls,
        client: OpsgenieNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> OpsgenieNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Opsgenie",
            config=OpsgenieNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OpsgenieNotificationChannelClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

NotificationChannel.register(OpsgenieNotificationChannelIntegration)
