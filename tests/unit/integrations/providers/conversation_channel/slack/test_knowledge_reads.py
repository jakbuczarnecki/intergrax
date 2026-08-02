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
_INVENTORY_TYPES = "public_channel,private_channel,im,mpim"


class _SlackResponse:
    def __init__(self, data: dict[str, Any], *, headers: dict[str, str] | None = None) -> None:
        self.data = data
        self.headers = headers or {}


class _SlackApiError(Exception):
    def __init__(self, code: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.response = _SlackResponse({"ok": False, "error": code}, headers=headers or {})


class _FakeWebClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("users_conversations", kwargs))
        if kwargs.get("cursor") == "inventory-page-2":
            return {
                "ok": True,
                "channels": [
                    {
                        "id": "C76543210",
                        "is_channel": True,
                        "is_private": True,
                        "name": "private-room",
                        "is_archived": False,
                    },
                    {
                        "id": "G01234567",
                        "is_mpim": True,
                        "name": "mpim-room",
                        "is_archived": False,
                    },
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
                },
                {
                    "id": "D01234567",
                    "is_im": True,
                    "user": "U22222222",
                    "is_archived": True,
                },
            ],
            "response_metadata": {"next_cursor": "inventory-page-2"},
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


def _reader(client: _FakeWebClient) -> SlackConversationKnowledgeReader:
    return SlackConversationKnowledgeReader(client)


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


async def test_inventory_uses_single_users_conversations_stream() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
    page = await reader.list_accessible_conversations_page(cursor=None, limit=50)
    inventory_calls = [call for call in client.calls if call[0] == "users_conversations"]
    assert len(inventory_calls) == 1
    inventory_call = inventory_calls[0][1]
    assert inventory_call["types"] == _INVENTORY_TYPES
    assert "user" not in inventory_call
    assert "token" not in inventory_call
    assert inventory_call["exclude_archived"] is False
    assert page.next_cursor == "inventory-page-2"
    kinds = {item.kind for item in page.items}
    assert SlackConversationKind.PUBLIC_CHANNEL in kinds
    assert SlackConversationKind.IM in kinds
    assert page.items[1].is_archived is True


async def test_inventory_paginates_provider_cursor_without_duplication() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
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
    inventory_calls = [call for call in client.calls if call[0] == "users_conversations"]
    assert len(inventory_calls) == 2
    assert inventory_calls[1][1]["cursor"] == "inventory-page-2"


async def test_inventory_uses_single_web_client() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
    await reader.list_accessible_conversations_page(cursor=None, limit=50)
    assert reader._web_client is client


async def test_inventory_duplicate_ids_fail_page() -> None:
    class _DuplicateInventoryClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "channels": [
                    {"id": _CONVERSATION_ID, "is_channel": True, "is_private": False},
                    {"id": _CONVERSATION_ID, "is_channel": True, "is_private": False},
                ],
            }

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_DuplicateInventoryClient()).list_accessible_conversations_page(
            cursor=None,
            limit=10,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_malformed_inventory_entry_fails_page() -> None:
    class _BadInventoryClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "channels": [{"id": ""}]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadInventoryClient()).list_accessible_conversations_page(cursor=None, limit=10)
    assert exc_info.value.slack_error == "malformed_response"


async def test_malformed_inventory_boolean_fails_page() -> None:
    class _BadBoolClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "channels": [{"id": _CONVERSATION_ID, "is_archived": "yes"}]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadBoolClient()).list_accessible_conversations_page(cursor=None, limit=10)
    assert exc_info.value.slack_error == "malformed_response"


async def test_history_public_channel_uses_bot_token_path() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
    await reader.read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=MAX_HISTORY_REPLY_PAGE_LIMIT,
        max_chars_per_message=1000,
    )
    history_call = next(call for call in client.calls if call[0] == "conversations_history")
    assert "token" not in history_call[1]


async def test_history_im_uses_bot_token_path() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
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


async def test_history_page_respects_boundaries_and_limit() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
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
    reader = _reader(client)
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
    reader = _reader(client)
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


async def test_exact_root_neighbor_is_malformed() -> None:
    class _NeighborRootClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("conversations_history", kwargs))
            if kwargs.get("oldest") == kwargs.get("latest") == _ROOT_TS:
                return {
                    "ok": True,
                    "messages": [
                        {
                            "ts": _EDITED_TS,
                            "user": "U111",
                            "text": "neighbor",
                        }
                    ],
                }
            return await super().conversations_history(**kwargs)

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_NeighborRootClient()).read_exact_message(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            message_ts=_ROOT_TS,
            root_thread_ts=None,
            window=_window(),
            expected_revision=None,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_exact_reply_first_page_target_only_no_root() -> None:
    client = _FakeWebClient()
    reader = _reader(client)
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
    reader = _reader(client)
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

    result = await _reader(_SingleReplyClient()).read_exact_message(
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
        await _reader(_NeighborClient()).read_exact_message(
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
        await _reader(_RepeatCursorClient()).read_exact_message(
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
        await _reader(_BadHistoryClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


@pytest.mark.parametrize(
    "message_payload",
    [
        {"ts": _ROOT_TS, "text": 123},
        {"ts": _ROOT_TS, "text": "ok", "reply_count": "3"},
        {"ts": _ROOT_TS, "text": "ok", "reply_count": -1},
        {"ts": _ROOT_TS, "text": "ok", "edited": []},
        {"ts": _ROOT_TS, "text": "ok", "is_starred": "yes"},
    ],
)
async def test_malformed_message_optional_fields_fail_page(message_payload: dict[str, Any]) -> None:
    class _BadMessageClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "messages": [message_payload]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadMessageClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_missing_text_allowed_for_subtype_messages() -> None:
    class _NoTextClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [{"ts": _ROOT_TS, "subtype": "bot_message", "user": "U111"}],
            }

    page = await _reader(_NoTextClient()).read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert page.items[0].text == ""


async def test_malformed_file_metadata_fails_page() -> None:
    class _BadFileClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _ROOT_TS,
                        "text": "with file",
                        "files": [{"id": "F001", "size": "big"}],
                    }
                ],
            }

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadFileClient()).read_conversation_history_page(
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
    reader = _reader(client)
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
        ("not_allowed_token_type", None),
        ("missing_scope", None),
        ("accesslimited", None),
        ("enterprise_is_restricted", None),
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
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            raise _SlackApiError(code, headers={"Retry-After": "3"})

    with pytest.raises((SlackConversationReadError, SlackConversationMessageNotFound)) as exc_info:
        await _reader(_ErrorClient()).read_conversation_history_page(
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
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            raise _SlackApiError("ratelimited", headers={"Retry-After": "3"})

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_RateLimitedClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "ratelimited"
    assert exc_info.value.retry_after_seconds == 3.0


async def test_absent_ok_field_is_malformed_response() -> None:
    class _NoOkClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {"messages": [{"ts": _ROOT_TS, "text": "x"}]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_NoOkClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


@pytest.mark.parametrize(
    "response_metadata",
    [
        [],
        "invalid",
        {"next_cursor": 123},
        {"next_cursor": []},
        {"next_cursor": " cursor"},
        {"next_cursor": "   "},
    ],
)
async def test_malformed_inventory_pagination_metadata_fails_page(
    response_metadata: object,
) -> None:
    class _BadMetadataClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "ok": True,
                "channels": [
                    {
                        "id": _CONVERSATION_ID,
                        "is_channel": True,
                        "is_private": False,
                        "name": "general",
                    }
                ],
                "response_metadata": response_metadata,
            }
            return payload

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadMetadataClient()).list_accessible_conversations_page(
            cursor=None,
            limit=10,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_inventory_pagination_metadata_absent_is_terminal() -> None:
    class _TerminalInventoryClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "channels": [
                    {
                        "id": _CONVERSATION_ID,
                        "is_channel": True,
                        "is_private": False,
                        "name": "general",
                    }
                ],
            }

    page = await _reader(_TerminalInventoryClient()).list_accessible_conversations_page(
        cursor=None,
        limit=10,
    )
    assert page.next_cursor is None


async def test_inventory_pagination_metadata_valid_cursor() -> None:
    class _ValidCursorClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "channels": [
                    {
                        "id": _CONVERSATION_ID,
                        "is_channel": True,
                        "is_private": False,
                        "name": "general",
                    }
                ],
                "response_metadata": {"next_cursor": "inventory-page-2"},
            }

    page = await _reader(_ValidCursorClient()).list_accessible_conversations_page(
        cursor=None,
        limit=10,
    )
    assert page.next_cursor == "inventory-page-2"


@pytest.mark.parametrize(
    "response_metadata",
    [
        [],
        "invalid",
        {"next_cursor": 123},
        {"next_cursor": []},
        {"next_cursor": " cursor"},
        {"next_cursor": "   "},
    ],
)
async def test_malformed_history_pagination_metadata_fails_page(
    response_metadata: object,
) -> None:
    class _BadHistoryMetadataClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "ok": True,
                "messages": [{"ts": _ROOT_TS, "user": "U111", "text": "root message"}],
                "response_metadata": response_metadata,
            }
            return payload

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadHistoryMetadataClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_history_pagination_empty_cursor_is_terminal() -> None:
    class _TerminalHistoryClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [{"ts": _ROOT_TS, "user": "U111", "text": "root message"}],
                "response_metadata": {"next_cursor": ""},
            }

    page = await _reader(_TerminalHistoryClient()).read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert page.next_cursor is None


@pytest.mark.parametrize(
    "thread_ts",
    [123, [], "", f" {_ROOT_TS}"],
)
async def test_malformed_thread_ts_fails_page(thread_ts: object) -> None:
    class _BadThreadClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [{"ts": _ROOT_TS, "thread_ts": thread_ts, "user": "U111", "text": "x"}],
            }

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadThreadClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_canonical_self_thread_root_normalizes_to_none() -> None:
    class _SelfThreadClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _ROOT_TS,
                        "thread_ts": _ROOT_TS,
                        "user": "U111",
                        "text": "root message",
                    }
                ],
            }

    page = await _reader(_SelfThreadClient()).read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert page.items[0].root_thread_ts is None


async def test_canonical_reply_thread_parses_root_thread_ts() -> None:
    class _ReplyThreadClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _REPLY_TS,
                        "thread_ts": _ROOT_TS,
                        "user": "U222",
                        "text": "reply body",
                    }
                ],
            }

    page = await _reader(_ReplyThreadClient()).read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    assert page.items[0].root_thread_ts == _ROOT_TS


@pytest.mark.parametrize(
    "edited",
    [
        {},
        {"ts": None},
        {"ts": 123},
        {"ts": ""},
        {"ts": " invalid "},
    ],
)
async def test_malformed_edited_marker_fails_page(edited: object) -> None:
    class _BadEditedClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [{"ts": _ROOT_TS, "user": "U111", "text": "ok", "edited": edited}],
            }

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadEditedClient()).read_conversation_history_page(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            window=_window(),
            cursor=None,
            limit=1,
            max_chars_per_message=1000,
        )
    assert exc_info.value.slack_error == "malformed_response"


async def test_valid_edited_marker_changes_revision() -> None:
    class _EditedClient(_FakeWebClient):
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _ROOT_TS,
                        "user": "U111",
                        "text": "edited body",
                        "edited": {"ts": "1704153700.000001"},
                    }
                ],
            }

    page = await _reader(_EditedClient()).read_conversation_history_page(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        window=_window(),
        cursor=None,
        limit=1,
        max_chars_per_message=1000,
    )
    message = page.items[0]
    assert message.edited_at is not None
    unedited = message.model_copy(update={"edited_at": None})
    assert compute_slack_conversation_message_revision(message) != (
        compute_slack_conversation_message_revision(unedited)
    )


@pytest.mark.parametrize(
    ("channel", "expected_kind"),
    [
        (
            {
                "id": _CONVERSATION_ID,
                "is_channel": True,
                "is_private": False,
                "name": "general",
            },
            SlackConversationKind.PUBLIC_CHANNEL,
        ),
        (
            {
                "id": "C76543210",
                "is_channel": True,
                "is_private": True,
                "name": "private-room",
            },
            SlackConversationKind.PRIVATE_CHANNEL,
        ),
        (
            {"id": "D01234567", "is_im": True, "user": "U22222222"},
            SlackConversationKind.IM,
        ),
        (
            {"id": "G01234567", "is_mpim": True, "name": "mpim-room"},
            SlackConversationKind.MPIM,
        ),
    ],
)
async def test_inventory_conversation_kind_shapes(
    channel: dict[str, Any],
    expected_kind: SlackConversationKind,
) -> None:
    class _KindClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "channels": [channel]}

    page = await _reader(_KindClient()).list_accessible_conversations_page(cursor=None, limit=10)
    assert page.items[0].kind is expected_kind


@pytest.mark.parametrize(
    "channel",
    [
        {"id": _CONVERSATION_ID},
        {"id": _CONVERSATION_ID, "is_channel": True, "is_im": True, "is_private": False},
        {"id": _CONVERSATION_ID, "is_channel": True, "is_mpim": True, "is_private": False},
        {"id": _CONVERSATION_ID, "is_im": True, "is_mpim": True},
        {"id": _CONVERSATION_ID, "is_channel": True, "is_private": "yes"},
        {"id": _CONVERSATION_ID, "is_channel": "yes", "is_private": False},
    ],
)
async def test_malformed_inventory_conversation_kind_fails_page(channel: dict[str, Any]) -> None:
    class _BadKindClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "channels": [channel]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadKindClient()).list_accessible_conversations_page(cursor=None, limit=10)
    assert exc_info.value.slack_error == "malformed_response"


@pytest.mark.parametrize(
    "channel_patch",
    [
        {"name": 123},
        {"user": 123, "is_im": True},
        {"topic": "not-a-mapping", "is_channel": True, "is_private": False},
        {"purpose": [], "is_channel": True, "is_private": False},
        {"topic": {"value": 123}, "is_channel": True, "is_private": False},
        {"purpose": {"value": 123}, "is_channel": True, "is_private": False},
    ],
)
async def test_malformed_inventory_summary_fields_fail_page(channel_patch: dict[str, Any]) -> None:
    base = {"id": _CONVERSATION_ID, "is_channel": True, "is_private": False}
    base.update(channel_patch)

    class _BadSummaryClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "channels": [base]}

    with pytest.raises(SlackConversationReadError) as exc_info:
        await _reader(_BadSummaryClient()).list_accessible_conversations_page(cursor=None, limit=10)
    assert exc_info.value.slack_error == "malformed_response"


async def test_inventory_empty_topic_and_purpose_normalize_to_none() -> None:
    class _EmptyTopicClient(_FakeWebClient):
        async def users_conversations(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "channels": [
                    {
                        "id": _CONVERSATION_ID,
                        "is_channel": True,
                        "is_private": False,
                        "name": "general",
                        "topic": {"value": ""},
                        "purpose": {"value": ""},
                    }
                ],
            }

    page = await _reader(_EmptyTopicClient()).list_accessible_conversations_page(
        cursor=None,
        limit=10,
    )
    assert page.items[0].safe_topic is None
    assert page.items[0].safe_purpose is None
