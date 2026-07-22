# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Chat conversation channel integration (contract-defined, runtime-unbound)."""

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

GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID = "google_chat"


class GoogleChatConversationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Google Chat conversation channel integration."""

    pass


class GoogleChatConversationChannelIntegration(ConversationChannelIntegrationContract):
    """
    Public Google Chat conversation channel entrypoint.

    Status: contract-defined, runtime-unbound. Vendor connectivity is not implemented.
    Inject a ``ConversationChannelBackend`` for contract tests or future runtime binding.
    """

    config: GoogleChatConversationChannelIntegrationConfig = GoogleChatConversationChannelIntegrationConfig()
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
            slug=GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
            default_detail="google_chat conversation channel ready probe",
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
    ) -> "GoogleChatConversationChannelIntegration":
        integration = cls.for_provider(
            provider_id=GOOGLE_CHAT_CONVERSATION_CHANNEL_PROVIDER_ID,
            display_name="Google Chat",
            config=GoogleChatConversationChannelIntegrationConfig(enabled=enabled),
        )
        integration._backend = backend
        return integration

    @property
    def backend(self) -> ConversationChannelBackend | None:
        return self._backend


ConversationChannelBackend.register(GoogleChatConversationChannelIntegration)
