# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Telegram conversation channel integration (contract-defined, runtime-unbound)."""

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import (
    ConversationChannelBackend,
    ConversationDeliveryReceipt,
    ConversationEventHandler,
    OutboundConversationMessage,
)
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.messaging import ConversationChannelIntegrationContract

TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID = "telegram"


class TelegramConversationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Telegram conversation channel integration."""

    pass


class TelegramConversationChannelIntegration(ConversationChannelIntegrationContract):
    """
    Public Telegram conversation channel entrypoint.

    Status: contract-defined, runtime-unbound. Vendor connectivity is not implemented.
    Inject a ``ConversationChannelBackend`` for contract tests or future runtime binding.
    """

    config: TelegramConversationChannelIntegrationConfig = TelegramConversationChannelIntegrationConfig()
    _backend: ConversationChannelBackend | None = PrivateAttr(default=None)

    async def start(self, handler: ConversationEventHandler) -> None:
        return await self._require_backend().start(handler)

    async def stop(self) -> None:
        return await self._require_backend().stop()

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        return await self._require_backend().send(message)

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_backend(),
            slug=TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID,
            default_detail="telegram conversation channel ready probe",
        )

    def _require_backend(self) -> ConversationChannelBackend:
        if self._backend is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires an injected ConversationChannelBackend",
            )
        return self._backend

    @classmethod
    def from_backend(
        cls,
        backend: ConversationChannelBackend,
        *,
        enabled: bool = False,
    ) -> "TelegramConversationChannelIntegration":
        integration = cls.for_provider(
            provider_id=TELEGRAM_CONVERSATION_CHANNEL_PROVIDER_ID,
            display_name="Telegram",
            config=TelegramConversationChannelIntegrationConfig(enabled=enabled),
        )
        integration._backend = backend
        return integration

    @property
    def backend(self) -> ConversationChannelBackend | None:
        return self._backend


ConversationChannelBackend.register(TelegramConversationChannelIntegration)
