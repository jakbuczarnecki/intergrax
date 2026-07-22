# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation channel integration (Socket Mode + Web API runtime)."""

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
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.runtime.integrations.categories.messaging import ConversationChannelIntegrationContract

SLACK_CONVERSATION_CHANNEL_PROVIDER_ID = "slack"


class SlackConversationChannelIntegration(ConversationChannelIntegrationContract):
    """
    Public Slack conversation channel entrypoint.

    Status: BETA — Socket Mode inbound + Web API outbound runtime binding supported.
    Inject a ``ConversationChannelBackend`` for tests, or construct via ``from_config``.
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
        config: SlackConversationChannelIntegrationConfig | None = None,
    ) -> SlackConversationChannelIntegration:
        resolved = config or SlackConversationChannelIntegrationConfig(enabled=enabled)
        integration = cls.for_provider(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            display_name="Slack",
            config=resolved,
        )
        integration._backend = backend
        return integration

    @classmethod
    def from_config(
        cls,
        config: SlackConversationChannelIntegrationConfig,
    ) -> SlackConversationChannelIntegration:
        """Construct production runtime when enabled; otherwise a disabled contract instance."""
        if not config.enabled:
            return cls.for_provider(
                provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                display_name="Slack",
                config=config,
            )
        backend = SlackConversationChannelBackend.from_config(config)
        return cls.from_backend(backend, enabled=True, config=config)

    @property
    def backend(self) -> ConversationChannelBackend | None:
        return self._backend


ConversationChannelBackend.register(SlackConversationChannelIntegration)

__all__ = [
    "SLACK_CONVERSATION_CHANNEL_PROVIDER_ID",
    "SlackConversationChannelIntegration",
    "SlackConversationChannelIntegrationConfig",
]
