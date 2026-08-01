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
    SlackConversationPointWindow,
    SlackConversationReadConfigurationError,
    SlackConversationReadError,
    SlackConversationSourceWindow,
    compare_slack_timestamps,
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
_REPLY_TS_PAGE2 = "1704153602.000001"
_EDITED_TS = "1704153603.000001"
_BOT_USER_ID = "U0BOTUSER"
_KNOWLEDGE_USER_TOKEN = "xoxp-test-knowledge-user-token"
_TEAM_ID = "T0WORKSPACE"
_OTHER_TEAM_ID = "T9OTHER"


class _SlackResponse:
    def __init__(self, data: dict[str, Any], *, headers: dict[str, str] | None = None) -> None:
        self.data = data
        self.headers = headers or {}


class _SlackApiError(Exception):
    def __init__(self, code: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.response = _SlackResponse({"ok": False, "error": code}, headers=headers or {})


class _FakeWebClient:
    def __init__(self, *, knowledge_team_id: str = _TEAM_ID) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._knowledge_team_id = knowledge_team_id

    async def auth_test(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("auth_test", kwargs))
        token = kwargs.get("token")
        if token == _KNOWLEDGE_USER_TOKEN:
            return {
                "ok": True,
                "user_id": "U0KNOWLEDGE",
                "team_id": self._knowledge_team_id,
            }
        return {"ok": True, "user_id": _BOT_USER_ID, "team_id": _TEAM_ID}

    async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("users_conversations", kwargs))
        types = kwargs.get("types", "")
        if types == "public_channel,private_channel":
            if kwargs.get("cursor") == "channels-page-2":
                return {
                    "ok": True,
                    "channels": [
                        {
                            "id": "C76543210",
                            "is_channel": True,
                            "is_private": True,
                            "name": "private-room",
                            "is_archived": False,
                        }
                    ],
                }
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
                    }
                ],
                "response_metadata": {"next_cursor": "channels-page-2"},
            }
        if kwargs.get("cursor") == "im-page-2":
            return {
                "ok": True,
                "channels": [
                    {
                        "id": "G01234567",
                        "is_mpim": True,
                        "name": "mpim-room",
                        "is_archived": False,
                    }
                ],
            }
        return {
            "ok": True,
            "channels": [
                {
                    "id": "D01234567",
                    "is_im": True,
                    "user": "U22222222",
                    "is_archived": False,
                }
            ],
            "response_metadata": {"next_cursor": "im-page-2"},
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
                    }
                ],
            }
        if kwargs.get("oldest") == kwargs.get("latest") == _ROOT_TS:
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _ROOT_TS,
                        "thread_ts": _ROOT_TS,
                        "user": "U111",
                        "text": "root message",
                        "reply_count": 1,
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
        if kwargs.get("cursor") == "reply-page-2":
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _REPLY_TS_PAGE2,
                        "user": "U333",
                        "text": "reply page two",
                        "thread_ts": _ROOT_TS,
                    }
                ],
            }
        if kwargs.get("oldest") == kwargs.get("latest") == _REPLY_TS_PAGE2:
            if kwargs.get("cursor") is None:
                return {
                    "ok": True,
                    "messages": [],
                    "response_metadata": {"next_cursor": "reply-page-2"},
                }
            if kwargs.get("cursor") == "reply-page-2":
                return {
                    "ok": True,
                    "messages": [
                        {
                            "ts": _REPLY_TS_PAGE2,
                            "user": "U333",
                            "text": "reply page two",
                            "thread_ts": _ROOT_TS,
                        }
                    ],
                }
        if kwargs.get("oldest") == kwargs.get("latest") == _REPLY_TS:
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _REPLY_TS,
                        "user": "U222",
                        "text": "target reply only",
                        "thread_ts": _ROOT_TS,
                    }
                ],
            }
        return {
            "ok": True,
            "messages": [
                {"ts": _ROOT_TS, "thread_ts": _ROOT_TS, "user": "U111", "text": "root message"},
                {"ts": _REPLY_TS, "user": "U222", "text": "reply body", "thread_ts": _ROOT_TS},
            ],
            "response_metadata": {"next_cursor": "reply-page-2"},
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


def _window() -> SlackConversationSourceWindow:
    return SlackConversationSourceWindow(oldest=_OLDEST, latest=_LATEST)


def _reader(
    client: _FakeWebClient,
    *,
    knowledge_user_token: str | None = None,
) -> SlackConversationKnowledgeReader:
    return SlackConversationKnowledgeReader(
        client,
        knowledge_user_token=knowledge_user_token,
    )


async def test_validate_slack_timestamp_rejects_aliases() -> None:
    with pytest.raises(ValueError):
        validate_slack_timestamp("1704067200.1")
    with pytest.raises(ValueError):
        validate_slack_timestamp(" 1704067200.000001")
    assert validate_slack_timestamp("1704067200.000001") == "1704067200.000001"


async def test_compare_slack_timestamps_exact_ordering() -> None:
    assert compare_slack_timestamps("1704067200.000001", "1706745600.000001") < 0
    assert compare_slack_timestamps("9999999999.999999", "10000000000.000001") < 0
    assert compare_slack_timestamps("1704067200.000001", "1704067200.000001") == 0


async def test_source_window_rejects_equal_and_reversed_boundaries() -> None:
    with pytest.raises(ValueError):
        SlackConversationSourceWindow(oldest=_OLDEST, latest=_OLDEST)
    with pytest.raises(ValueError):
        SlackConversationSourceWindow(oldest=_LATEST, latest=_OLDEST)
    SlackConversationSourceWindow(oldest=_OLDEST, latest=_LATEST)


async def test_point_window_allows_exact_timestamp() -> None:
    point = SlackConversationPointWindow(message_ts=_ROOT_TS)
    assert point.oldest == _ROOT_TS
    assert point.latest == _ROOT_TS


async def test_inventory_without_knowledge_token_exposes_im_mpim_only() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
    page = await reader.list_accessible_conversations_page(cursor=None, limit=50)
    inventory_call = next(call for call in client.calls if call[0] == "users_conversations")
    assert "user" not in inventory_call[1]
    assert inventory_call[1]["types"] == "im,mpim"
    assert all(
        item.kind in {SlackConversationKind.IM, SlackConversationKind.MPIM}
        for item in page.items
    )
    assert page.next_cursor == "im-page-2"


async def test_inventory_with_knowledge_token_uses_user_token_for_channels() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    page = await reader.list_accessible_conversations_page(cursor=None, limit=50)
    channel_call = next(call for call in client.calls if call[0] == "users_conversations")
    assert channel_call[1]["types"] == "public_channel,private_channel"
    assert channel_call[1]["token"] == _KNOWLEDGE_USER_TOKEN
    assert page.items[0].kind is SlackConversationKind.PUBLIC_CHANNEL
    assert page.next_cursor is not None


async def test_inventory_dual_stream_paginates_without_duplication() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    seen_ids: set[str] = set()
    cursor: str | None = None
    while True:
        page = await reader.list_accessible_conversations_page(cursor=cursor, limit=50)
        for item in page.items:
            assert item.conversation_id not in seen_ids
            seen_ids.add(item.conversation_id)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert _CONVERSATION_ID in seen_ids
    assert "D01234567" in seen_ids
    assert "G01234567" in seen_ids
    assert "C76543210" in seen_ids


async def test_inventory_uses_single_web_client() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    await reader.list_accessible_conversations_page(cursor=None, limit=50)
    assert reader._web_client is client


async def test_workspace_identity_validation_same_team() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert reader._bot_team_id == _TEAM_ID
    assert reader._knowledge_team_id == _TEAM_ID


async def test_workspace_mismatch_fails_safely() -> None:
    client = _FakeWebClient(knowledge_team_id=_OTHER_TEAM_ID)
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    with pytest.raises(SlackConversationReadConfigurationError):
        await reader.read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )


async def test_tokens_never_appear_in_errors() -> None:
    client = _FakeWebClient(knowledge_team_id=_OTHER_TEAM_ID)
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    with pytest.raises(SlackConversationReadConfigurationError) as exc_info:
        await reader.read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert _KNOWLEDGE_USER_TOKEN not in str(exc_info.value)


async def test_tokens_never_appear_in_inventory_models() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    page = await reader.list_accessible_conversations_page(cursor=None, limit=50)
    assert _KNOWLEDGE_USER_TOKEN not in repr(page.items)


async def test_public_channel_without_user_token_fails_before_provider() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
    with pytest.raises(SlackConversationReadConfigurationError):
        await reader.read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert not any(call[0] == "conversations_history" for call in client.calls)


async def test_history_public_channel_uses_user_token_override() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    history_call = next(call for call in client.calls if call[0] == "conversations_history")
    assert history_call[1]["token"] == _KNOWLEDGE_USER_TOKEN


async def test_history_im_uses_bot_token_path() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    await reader.read_conversation_history_page(
        conversation_id="D01234567",
        conversation_kind=SlackConversationKind.IM,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    history_call = next(call for call in client.calls if call[0] == "conversations_history")
    assert "token" not in history_call[1]


async def test_malformed_inventory_entry_fails_page() -> None:
    class _BadInventoryClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "channels": [{"id": ""}]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadInventoryClient()).list_accessible_conversations_page(cursor=None, limit=10)
    assert exc_info.value.slack_error == "malformed_response"


async def test_history_page_respects_boundaries_and_limit() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    page = await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    history_call = next(call for call in client.calls if call[0] == "conversations_history")
    assert history_call[1]["oldest"] == _OLDEST
    assert history_call[1]["latest"] == _LATEST
    assert history_call[1]["limit"] == MAX_HISTORY_REPLY_PAGE_LIMIT
    assert page.items[0].message_ts == _ROOT_TS


async def test_thread_replies_page_normalizes_real_root_shape_and_deduplicates() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    page = await reader.read_thread_replies_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        root_message_ts=_ROOT_TS,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    assert [item.message_ts for item in page.items] == [_REPLY_TS]
    assert all(item.root_thread_ts == _ROOT_TS for item in page.items)


async def test_exact_message_lookup_for_root_and_reply() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    root = await reader.read_exact_message(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        message_ts=_ROOT_TS,
        root_thread_ts=None,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert root.found is True
    assert root.message is not None
    assert root.message.root_thread_ts is None
    reply = await reader.read_exact_message(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        message_ts=_REPLY_TS,
        root_thread_ts=_ROOT_TS,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert reply.found is True
    assert reply.message is not None
    assert reply.message.root_thread_ts == _ROOT_TS


async def test_exact_reply_first_page_target_only_no_root() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    result = await reader.read_exact_message(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        message_ts=_REPLY_TS,
        root_thread_ts=_ROOT_TS,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert result.found is True
    assert result.message is not None
    assert result.message.message_ts == _REPLY_TS


async def test_exact_reply_first_page_empty_with_next_cursor() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    result = await reader.read_exact_message(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        message_ts=_REPLY_TS_PAGE2,
        root_thread_ts=_ROOT_TS,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert result.found is True
    assert result.message is not None
    assert result.message.message_ts == _REPLY_TS_PAGE2


async def test_exact_reply_lookup_not_found_after_pages_exhausted() -> None:
    missing_ts = "1704999999.000001"

    class _SingleReplyClient(_FakeWebClient):
        async def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("conversations_replies", kwargs))
            if kwargs.get("oldest") == kwargs.get("latest") == missing_ts:
                return {"ok": True, "messages": []}
            return {
                "ok": True,
                "messages": [
                    {"ts": _ROOT_TS, "thread_ts": _ROOT_TS, "user": "U111", "text": "root"},
                    {"ts": _REPLY_TS, "user": "U222", "text": "only reply", "thread_ts": _ROOT_TS},
                ],
            }

    result = await _reader(
        _SingleReplyClient(),
        knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
    ).read_exact_message(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        message_ts=missing_ts,
        root_thread_ts=_ROOT_TS,
        window=_window(),
        expected_revision=None,
        max_chars_per_message=1000,
    )
    assert result.found is False


async def test_exact_reply_neighboring_message_is_malformed() -> None:
    class _NeighborClient(_FakeWebClient):
        async def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("conversations_replies", kwargs))
            if kwargs.get("oldest") == kwargs.get("latest") == _REPLY_TS_PAGE2:
                return {
                    "ok": True,
                    "messages": [
                        {
                            "ts": _REPLY_TS,
                            "user": "U222",
                            "text": "neighbor",
                            "thread_ts": _ROOT_TS,
                        }
                    ],
                }
            return await super().conversations_replies(**kwargs)

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(
            _NeighborClient(),
            knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
        ).read_exact_message(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            message_ts=_REPLY_TS_PAGE2,
            root_thread_ts=_ROOT_TS,
            window=_window(),
            expected_revision=None,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_exact_reply_repeated_cursor_is_malformed() -> None:
    class _RepeatCursorClient(_FakeWebClient):
        async def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("conversations_replies", kwargs))
            if kwargs.get("oldest") == kwargs.get("latest") == _REPLY_TS_PAGE2:
                return {
                    "ok": True,
                    "messages": [],
                    "response_metadata": {"next_cursor": "repeat-cursor"},
                }
            return await super().conversations_replies(**kwargs)

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(
            _RepeatCursorClient(),
            knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
        ).read_exact_message(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            message_ts=_REPLY_TS_PAGE2,
            root_thread_ts=_ROOT_TS,
            window=_window(),
            expected_revision=None,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_malformed_history_entry_fails_page() -> None:
    class _BadHistoryClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "messages": [{"ts": "bad-ts", "text": "x"}]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(
            _BadHistoryClient(),
            knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
        ).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_revision_changes_when_text_changes() -> None:
    client = _FakeWebClient()
    reader = _reader(client, knowledge_user_token=_KNOWLEDGE_USER_TOKEN)
    page = await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
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
        knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
    )
    backend = SlackConversationChannelBackend(config=config, web_client=client)
    integration = SlackConversationChannelIntegration.from_backend(backend, enabled=True, config=config)
    page = await integration.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert page.items[0].message_ts == _ROOT_TS
    assert any(call[0] == "conversations_history" for call in client.calls)


@pytest.mark.parametrize(
    ("code", "expected_retry_after"),
    [
        ("ratelimited", 3.0),
        ("invalid_auth", None),
        ("missing_scope", None),
        ("token_expired", None),
        ("team_access_not_granted", None),
        ("channel_not_found", None),
    ],
)
async def test_slack_api_error_matrix(code: str, expected_retry_after: float | None) -> None:
    from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.errors import (
        SlackConversationMessageNotFound,
    )

    class _ErrorClient:
        async def auth_test(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "user_id": _BOT_USER_ID, "team_id": _TEAM_ID}

        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            raise _SlackApiError(code, headers={"Retry-After": "3"})

    with pytest.raises((SlackConversationReadError, SlackConversationMessageNotFound)) as exc_info:
        await _reader(
            _ErrorClient(),
            knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
        ).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    if isinstance(exc_info.value, SlackConversationReadError):
        assert exc_info.value.slack_error == code
        if code == "ratelimited":
            assert exc_info.value.retry_after_seconds == expected_retry_after


async def test_ratelimited_error_uses_slack_response_shape() -> None:
    class _RateLimitedClient:
        async def auth_test(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "user_id": _BOT_USER_ID, "team_id": _TEAM_ID}

        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            raise _SlackApiError("ratelimited", headers={"Retry-After": "3"})

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(
            _RateLimitedClient(),
            knowledge_user_token=_KNOWLEDGE_USER_TOKEN,
        ).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "ratelimited"
    assert exc_info.value.retry_after_seconds == 3.0
