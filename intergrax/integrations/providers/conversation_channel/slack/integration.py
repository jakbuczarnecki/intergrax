# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation channel integration (Socket Mode + Web API runtime)."""

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationConfigurationError,
)
from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentContent,
    ConversationAttachmentFetcher,
    ConversationAttachmentFetchError,
    ConversationAttachmentReference,
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
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    DEFAULT_MESSAGE_MAX_CHARS,
    SlackConversationExactMessageResult,
    SlackConversationFileReference,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationKnowledgeReadClient,
    SlackConversationMessagePage,
    SlackConversationReadConfigurationError,
    SlackConversationSourceWindow,
)
from intergrax.runtime.integrations.categories.messaging import (
    ConversationChannelIntegrationContract,
)

SLACK_CONVERSATION_CHANNEL_PROVIDER_ID = "slack"


class SlackConversationChannelIntegration(ConversationChannelIntegrationContract):
    """
    Public Slack conversation channel entrypoint.

    Status: BETA — Socket Mode inbound + Web API outbound runtime binding supported.
    Inject a ``ConversationChannelBackend`` for tests, or construct via ``from_config``.
    """

    config: SlackConversationChannelIntegrationConfig = (
        SlackConversationChannelIntegrationConfig()
    )
    _backend: ConversationChannelBackend | None = PrivateAttr(default=None)

    async def start(self, handler: ConversationEventHandler) -> None:
        return await self._require_backend().start(handler)

    async def stop(self) -> None:
        return await self._require_backend().stop()

    async def send(
        self, message: OutboundConversationMessage
    ) -> ConversationDeliveryReceipt:
        return await self._require_backend().send(message)

    async def fetch_attachment(
        self,
        attachment: ConversationAttachmentReference,
        *,
        max_bytes: int,
    ) -> ConversationAttachmentContent:
        backend = self._require_backend()
        if not isinstance(backend, ConversationAttachmentFetcher):
            raise ConversationAttachmentFetchError(kind="attachment_fetch_unavailable")
        return await backend.fetch_attachment(attachment, max_bytes=max_bytes)

    async def list_accessible_conversations_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> SlackConversationInventoryPage:
        return await self._require_knowledge_read_client().list_accessible_conversations_page(
            cursor=cursor,
            limit=limit,
        )

    async def read_conversation_history_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        window: SlackConversationSourceWindow,
        cursor: str | None,
        limit: int,
        max_chars_per_message: int = DEFAULT_MESSAGE_MAX_CHARS,
    ) -> SlackConversationMessagePage:
        return (
            await self._require_knowledge_read_client().read_conversation_history_page(
                conversation_id=conversation_id,
                conversation_kind=conversation_kind,
                window=window,
                cursor=cursor,
                limit=limit,
                max_chars_per_message=max_chars_per_message,
            )
        )

    async def read_recent_conversation_messages_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        window: SlackConversationSourceWindow,
        limit: int,
        max_chars_per_message: int = DEFAULT_MESSAGE_MAX_CHARS,
        cursor: str | None = None,
    ) -> SlackConversationMessagePage:
        return await self._require_knowledge_read_client().read_recent_conversation_messages_page(
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,
            window=window,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
            cursor=cursor,
        )

    async def read_thread_replies_page(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        root_message_ts: str,
        window: SlackConversationSourceWindow,
        cursor: str | None,
        limit: int,
        max_chars_per_message: int = DEFAULT_MESSAGE_MAX_CHARS,
    ) -> SlackConversationMessagePage:
        return await self._require_knowledge_read_client().read_thread_replies_page(
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,
            root_message_ts=root_message_ts,
            window=window,
            cursor=cursor,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )

    async def read_exact_message(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        message_ts: str,
        root_thread_ts: str | None,
        window: SlackConversationSourceWindow,
        expected_revision: str | None = None,
        max_chars_per_message: int = DEFAULT_MESSAGE_MAX_CHARS,
    ) -> SlackConversationExactMessageResult:
        return await self._require_knowledge_read_client().read_exact_message(
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,
            message_ts=message_ts,
            root_thread_ts=root_thread_ts,
            window=window,
            expected_revision=expected_revision,
            max_chars_per_message=max_chars_per_message,
        )

    async def read_file_info(
        self,
        *,
        file_id: str,
        conversation_kind: SlackConversationKind | None = None,
    ) -> SlackConversationFileReference:
        return await self._require_knowledge_read_client().read_file_info(
            file_id=file_id,
            conversation_kind=conversation_kind,
        )

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

    def _require_knowledge_read_client(self) -> SlackConversationKnowledgeReadClient:
        backend = self._require_backend()
        if not isinstance(backend, SlackConversationKnowledgeReadClient):
            raise SlackConversationReadConfigurationError(
                "Slack conversation knowledge read requires SlackConversationChannelBackend",
            )
        return backend

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
