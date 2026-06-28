# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sqs message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SQS_MESSAGE_BUS_PROVIDER_ID = "sqs"


class SqsMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Sqs message bus integration."""

    pass


@runtime_checkable
class SqsMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SqsMessageBusIntegration(MessageBusIntegrationContract):
    """
    Sqs message bus integration.

    The legacy facade (create_sqs_message_bus) remains separate and backward-compatible.
    """

    config: SqsMessageBusIntegrationConfig = SqsMessageBusIntegrationConfig()
    _client: SqsMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SqsMessageBusClient,
        *,
        enabled: bool = False,
    ) -> SqsMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=SQS_MESSAGE_BUS_PROVIDER_ID,
            display_name="Sqs",
            config=SqsMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SqsMessageBusClient | None:
        return self._client
