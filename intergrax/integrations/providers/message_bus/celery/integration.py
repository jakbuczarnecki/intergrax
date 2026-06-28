# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Celery message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CELERY_MESSAGE_BUS_PROVIDER_ID = "celery"


class CeleryMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Celery message bus integration."""

    pass


@runtime_checkable
class CeleryMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CeleryMessageBusIntegration(MessageBusIntegrationContract):
    """
    Celery message bus integration.

    The legacy facade (create_celery_integration) remains separate and backward-compatible.
    """

    config: CeleryMessageBusIntegrationConfig = CeleryMessageBusIntegrationConfig()
    _client: CeleryMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: CeleryMessageBusClient,
        *,
        enabled: bool = False,
    ) -> CeleryMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=CELERY_MESSAGE_BUS_PROVIDER_ID,
            display_name="Celery",
            config=CeleryMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CeleryMessageBusClient | None:
        return self._client
