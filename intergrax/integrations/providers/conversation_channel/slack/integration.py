# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation channel integration (contract-defined, runtime-unbound)."""

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

SLACK_CONVERSATION_CHANNEL_PROVIDER_ID = "slack"


class SlackConversationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Slack conversation channel integration."""

    pass


class SlackConversationChannelIntegration(ConversationChannelIntegrationContract):
    """
    Public Slack conversation channel entrypoint.

    Status: contract-defined, runtime-unbound. Vendor connectivity is not implemented.
    Inject a ``ConversationChannelBackend`` for contract tests or future runtime binding.
    """

    config: SlackConversationChannelIntegrationConfig = SlackConversationChannelIntegrationConfig()
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
            slug=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            default_detail="slack conversation channel ready probe",
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
    ) -> "SlackConversationChannelIntegration":
        integration = cls.for_provider(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            display_name="Slack",
            config=SlackConversationChannelIntegrationConfig(enabled=enabled),
        )
        integration._backend = backend
        return integration

    @property
    def backend(self) -> ConversationChannelBackend | None:
        return self._backend


ConversationChannelBackend.register(SlackConversationChannelIntegration)
