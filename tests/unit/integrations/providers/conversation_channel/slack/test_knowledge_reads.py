# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Slack conversation knowledge-read primitives."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    MAX_HISTORY_REPLY_PAGE_LIMIT,
    SlackConversationKind,
    SlackConversationReadError,
    SlackConversationSourceWindow,
    validate_slack_timestamp,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.reader import (
    SlackConversationKnowledgeReader,
    compute_slack_conversation_message_revision,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_CONVERSATION_ID = "C01234567"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_REPLY_TS = "1704153601.000001"
_EDITED_TS = "1704153602.000001"


class _FakeWebClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def conversations_list(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("conversations_list", kwargs))
        return {
            "ok": True,
            "channels": [
                {
                    "id": _CONVERSATION_ID,
                    "name": "general",
                    "is_channel": True,
                    "is_private": False,
                    "is_archived": False,
                    "created": 1704067200,
                    "topic": {"value": "Team updates"},
                    "purpose": {"value": "Announcements"},
                }
            ],
            "response_metadata": {"next_cursor": "cursor-page-2"},
        }

    async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("conversations_history", kwargs))
        if kwargs.get("cursor") == "history-page-2":
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _EDITED_TS,
                        "user": "U111",
                        "text": "edited body",
                        "edited": {"ts": "1704153700.000001"},
                        "files": [
                            {
                                "id": "F001",
                                "name": "notes.txt",
                                "title": "Notes",
                                "mimetype": "text/plain",
                                "filetype": "text",
                                "size": 12,
                                "mode": "hosted",
                                "created": 1704153600,
                                "is_external": False,
                            }
                        ],
                    }
                ],
            }
        return {
            "ok": True,
            "messages": [
                {
                    "ts": _ROOT_TS,
                    "user": "U111",
                    "text": "root message",
                    "reply_count": 1,
                }
            ],
            "response_metadata": {"next_cursor": "history-page-2"},
        }

    async def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("conversations_replies", kwargs))
        return {
            "ok": True,
            "messages": [
                {"ts": _ROOT_TS, "user": "U111", "text": "root message", "reply_count": 1},
                {"ts": _REPLY_TS, "user": "U222", "text": "reply body", "thread_ts": _ROOT_TS},
            ],
        }

    async def files_info(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("files_info", kwargs))
        return {
            "ok": True,
            "file": {
                "id": "F001",
                "name": "notes.txt",
                "title": "Notes",
                "mimetype": "text/plain",
                "filetype": "text",
                "size": 12,
                "mode": "hosted",
                "created": 1704153600,
                "is_external": False,
            },
        }


class _SlackApiError(Exception):
    def __init__(self, code: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.response = {"error": code, "headers": headers or {}}


def _window() -> SlackConversationSourceWindow:
    return SlackConversationSourceWindow(oldest=_OLDEST, latest=_LATEST)


async def test_validate_slack_timestamp_rejects_aliases() -> None:
    with pytest.raises(ValueError):
        validate_slack_timestamp("1704067200.1")
    with pytest.raises(ValueError):
        validate_slack_timestamp(" 1704067200.000001")
    assert validate_slack_timestamp("1704067200.000001") == "1704067200.000001"


async def test_list_accessible_conversations_page_maps_safe_inventory() -> None:
    client = _FakeWebClient()
    reader = SlackConversationKnowledgeReader(client)
    page = await reader.list_accessible_conversations_page(cursor=None, limit=50)
    assert page.items[0].conversation_id == _CONVERSATION_ID
    assert page.items[0].kind is SlackConversationKind.PUBLIC_CHANNEL
    assert page.items[0].safe_topic == "Team updates"
    assert page.next_cursor == "cursor-page-2"


async def test_history_page_respects_boundaries_and_limit() -> None:
    client = _FakeWebClient()
    reader = SlackConversationKnowledgeReader(client)
    page = await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    history_call = client.calls[0]
    assert history_call[0] == "conversations_history"
    assert history_call[1]["oldest"] == _OLDEST
    assert history_call[1]["latest"] == _LATEST
    assert history_call[1]["limit"] == MAX_HISTORY_REPLY_PAGE_LIMIT
    assert page.items[0].message_ts == _ROOT_TS


async def test_thread_replies_page_deduplicates_root() -> None:
    client = _FakeWebClient()
    reader = SlackConversationKnowledgeReader(client)
    page = await reader.read_thread_replies_page(
        conversation_id=_CONVERSATION_ID,
        root_message_ts=_ROOT_TS,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    assert [item.message_ts for item in page.items] == [_REPLY_TS]


async def test_exact_message_lookup_for_root_and_reply() -> None:
    client = _FakeWebClient()
    reader = SlackConversationKnowledgeReader(client)
    root = await reader.read_exact_message(
        conversation_id=_CONVERSATION_ID,
        message_ts=_ROOT_TS,
        root_thread_ts=None,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert root.found is True
    assert root.message is not None
    reply = await reader.read_exact_message(
        conversation_id=_CONVERSATION_ID,
        message_ts=_REPLY_TS,
        root_thread_ts=_ROOT_TS,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert reply.found is True
    assert reply.message is not None
    assert reply.message.root_thread_ts == _ROOT_TS


async def test_revision_changes_when_text_changes() -> None:
    client = _FakeWebClient()
    reader = SlackConversationKnowledgeReader(client)
    page = await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        window=_window(),
        cursor="history-page-2",
        limit=1,
        max_chars_per_message=1000,
    )
    message = page.items[0]
    first_revision = compute_slack_conversation_message_revision(message)
    edited = message.model_copy(update={"text": "changed"})
    second_revision = compute_slack_conversation_message_revision(edited)
    assert first_revision != second_revision


async def test_backend_reuses_injected_web_client_for_knowledge_reads() -> None:
    client = _FakeWebClient()
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-token-value",
        bot_token="xoxb-test-token-value",
    )
    backend = SlackConversationChannelBackend(config=config, web_client=client)
    integration = SlackConversationChannelIntegration.from_backend(backend, enabled=True, config=config)
    page = await integration.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert page.items[0].message_ts == _ROOT_TS
    assert any(call[0] == "conversations_history" for call in client.calls)


async def test_ratelimited_error_is_normalized_without_raw_headers() -> None:
    class _RateLimitedClient:
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            raise _SlackApiError("ratelimited", headers={"Retry-After": "3"})

    with pytest.raises(SlackConversationReadError) as exc_info:
        await SlackConversationKnowledgeReader(_RateLimitedClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "ratelimited"
    assert exc_info.value.retry_after_seconds == 3.0
