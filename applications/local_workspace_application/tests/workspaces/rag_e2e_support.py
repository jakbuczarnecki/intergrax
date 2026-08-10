# © Artur Czarnecki. All rights reserved.

"""Shared test-only support for local-workspace RAG application E2Es."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationExactMessageResult,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    SlackConversationSummary,
    compute_slack_conversation_message_revision,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    parse_slack_ts,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

_MARKER_ROOT = "deployment of project Atlas failed because of database timeout"
_MARKER_REPLY = "connection pool exhaustion"
_MARKER_EDIT = "increase pool and add alerting"
_ATLAS_INVESTIGATION = "Atlas deployment investigation identified the database timeout"
_ATLAS_MITIGATION = "Atlas deployment mitigation is to increase the database connection pool"
_UNRELATED_MESSAGE = "routine social update: lunch is at noon"
_CONVERSATION_ID = "C01234567"
_CONNECTION = "conn.slack"
_GRAPH_MAILBOX = "user-abc-123"
_GRAPH_TEAM_ID = "team-abc-123"
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_ROOT_2_TS = "1704153602.000001"
_REPLY_TS = "1704153601.000001"
_REPLY_2_TS = "1704153601.000002"
_REPLY_3_TS = "1704153601.000003"
_REPLY_4_TS = "1704153601.000004"
_TS = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_PREFIX = "/v1/local_workspace"
_SIGNING_KEY = "e2e-connected-source-signing-key"


class _RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text
        self.messages: list[tuple[tuple[str, str], ...]] = []

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id
        self.messages.append(tuple((message.role, message.content) for message in messages))
        return build_adapter_response(content=self._fixed_text)


class _RecordingSecretsStore:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


def _message(
    *,
    message_ts: str,
    text: str,
    reply_count: int = 0,
    root_thread_ts: str | None = None,
    edited_at: datetime | None = None,
) -> SlackConversationMessage:
    created_at = parse_slack_ts(message_ts) or _TS
    return SlackConversationMessage(
        conversation_id=_CONVERSATION_ID,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id="U111",
        text=text,
        subtype=None,
        created_at=created_at,
        edited_at=edited_at,
        reply_count=reply_count,
        files=(),
        provider_metadata={},
    )


class _SlackFakeBackend(SlackConversationChannelBackend):
    def __init__(self) -> None:
        self.history_calls = 0
        self.reply_calls = 0
        self._history_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_ROOT_TS,
                        text=_MARKER_ROOT,
                        reply_count=4,
                    ),
                    _message(
                        message_ts=_ROOT_2_TS,
                        text=_UNRELATED_MESSAGE,
                    ),
                ),
                next_cursor="history-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(),
            ),
        ]
        self._reply_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_REPLY_TS,
                        text=f"{_ATLAS_INVESTIGATION}; {_MARKER_REPLY}",
                        root_thread_ts=_ROOT_TS,
                    ),
                    _message(
                        message_ts=_REPLY_2_TS,
                        text="identified connection pool exhaustion",
                        root_thread_ts=_ROOT_TS,
                    ),
                ),
                next_cursor="replies-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_REPLY_3_TS,
                        text=_ATLAS_MITIGATION,
                        root_thread_ts=_ROOT_TS,
                    ),
                    _message(
                        message_ts=_REPLY_4_TS,
                        text=f"Atlas deployment final decision: {_MARKER_EDIT}",
                        root_thread_ts=_ROOT_TS,
                    ),
                ),
            ),
        ]
        self._content: dict[str, SlackConversationMessage] = {}

    async def list_accessible_conversations_page(self, *, cursor, limit):
        return SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id=_CONVERSATION_ID,
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            ),
            next_cursor=None,
        )

    async def read_conversation_history_page(
        self, **kwargs: Any
    ) -> SlackConversationMessagePage:
        self.history_calls += 1
        cursor = kwargs.get("cursor")
        page = self._history_pages[1] if cursor == "history-2" else self._history_pages[0]
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_thread_replies_page(
        self, **kwargs: Any
    ) -> SlackConversationMessagePage:
        self.reply_calls += 1
        cursor = kwargs.get("cursor")
        page = self._reply_pages[1] if cursor == "replies-2" else self._reply_pages[0]
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_exact_message(self, **kwargs: Any) -> SlackConversationExactMessageResult:
        message_ts = kwargs["message_ts"]
        message = self._content.get(message_ts)
        if message is None:
            message = _message(message_ts=message_ts, text="exact")
        revision = kwargs.get("expected_revision")
        if revision is not None and revision != compute_slack_conversation_message_revision(message):
            from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
                SlackConversationMessageChanged,
            )

            raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=message)

    async def read_file_info(self, **kwargs: Any):
        raise NotImplementedError
