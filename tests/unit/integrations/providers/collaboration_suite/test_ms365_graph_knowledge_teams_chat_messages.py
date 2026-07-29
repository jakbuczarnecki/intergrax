# © Artur Czarnecki. All rights reserved.

"""Unit tests for Microsoft Graph Teams Chat knowledge-read messages surface."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    parse_msgraph_teams_chat,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageState,
    MsGraphTeamsChatMessageWindow,
    MsGraphTeamsChatMessagesReader,
    format_msgraph_teams_chat_window_datetime,
    parse_msgraph_teams_chat_message,
    validate_msgraph_teams_chat_messages_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX = "user@contoso.com"
_CHAT_ID = "19:chat-abc@thread.v2"
_MESSAGE_ID = "msg-001"
_ETAG = "etag-1"
_QUOTED_MAILBOX = quote(_MAILBOX, safe="")
_QUOTED_CHAT = quote(_CHAT_ID, safe="")
_PREFER = {"Prefer": "include-unknown-enum-members"}
_SAFE = "unexpected Microsoft Graph Teams chat messages response"
_CONT = "invalid Microsoft Graph Teams chat messages continuation"


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _chat():
    return parse_msgraph_teams_chat(
        {
            "id": _CHAT_ID,
            "chatType": "group",
            "createdDateTime": "2024-01-01T10:00:00Z",
            "lastUpdatedDateTime": "2024-01-02T10:00:00Z",
            "isHiddenForAllMembers": False,
        },
        expected_mailbox_user_id=_MAILBOX,
    )


def _active_message_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": _MESSAGE_ID,
        "chatId": _CHAT_ID,
        "etag": _ETAG,
        "messageType": "message",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-01T11:00:00Z",
        "deletedDateTime": None,
        "importance": "normal",
        "body": {"contentType": "text", "content": "Hello"},
        "from": {"user": {"id": "u1", "displayName": "Alice"}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base


def _window() -> MsGraphTeamsChatMessageWindow:
    return MsGraphTeamsChatMessageWindow(
        start_at=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        end_at=datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc),
    )


def test_format_window_datetime_utc_z() -> None:
    value = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
    assert format_msgraph_teams_chat_window_datetime(value) == "2024-01-01T12:30:45Z"


def test_window_rejects_naive_and_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        MsGraphTeamsChatMessageWindow(
            start_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )


def test_parse_active_text_message() -> None:
    msg = parse_msgraph_teams_chat_message(
        _active_message_payload(),
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert msg.state is MsGraphTeamsChatMessageState.ACTIVE
    assert msg.body_content == "Hello"
    assert msg.sender is not None
    assert msg.sender.display_name == "Alice"


def test_parse_deleted_message() -> None:
    msg = parse_msgraph_teams_chat_message(
        {
            "id": _MESSAGE_ID,
            "chatId": _CHAT_ID,
            "etag": _ETAG,
            "messageType": "message",
            "createdDateTime": "2024-01-01T10:00:00Z",
            "lastModifiedDateTime": "2024-01-01T12:00:00Z",
            "deletedDateTime": "2024-01-01T12:00:00Z",
            "importance": "normal",
        },
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert msg.state is MsGraphTeamsChatMessageState.DELETED
    assert msg.deleted_at is not None
    assert msg.body_kind is None


def test_parse_reference_attachment_https_only() -> None:
    msg = parse_msgraph_teams_chat_message(
        _active_message_payload(
            attachments=[
                {
                    "id": "att-1",
                    "contentType": "reference",
                    "contentUrl": "https://contoso.sharepoint.com/file",
                }
            ]
        ),
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert msg.attachments[0].content_url is not None
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_chat_message(
            _active_message_payload(
                attachments=[
                    {
                        "id": "att-1",
                        "contentType": "reference",
                        "contentUrl": "http://insecure.example/file",
                    }
                ]
            ),
            expected_mailbox_user_id=_MAILBOX,
            expected_chat_id=_CHAT_ID,
            max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )


def test_parse_channel_identity_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_chat_message(
            _active_message_payload(channelIdentity={"teamId": "t"}),
            expected_mailbox_user_id=_MAILBOX,
            expected_chat_id=_CHAT_ID,
            max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )


def test_messages_continuation_validation() -> None:
    url = (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
        f"{_QUOTED_CHAT}/messages?$skiptoken=x"
    )
    cont = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    validate_msgraph_teams_chat_messages_continuation(
        cont, mailbox_user_id=_MAILBOX, chat_id=_CHAT_ID, graph_base_url=_GRAPH_BASE
    )


def _reader(http: MagicMock) -> MsGraphTeamsChatMessagesReader:
    return MsGraphTeamsChatMessagesReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def test_reader_snapshot_request_params() -> None:
    http = MagicMock()
    http.get.return_value = MagicMock(status_code=200, json=lambda: {"value": []})
    window = _window()
    _reader(http).read_messages_snapshot_page(
        chat=_chat(),
        window=window,
        continuation=None,
        limit=25,
        max_chars_per_message=1000,
    )
    _, kwargs = http.get.call_args
    params = kwargs["params"]
    assert params["$top"] == 25
    assert params["$orderby"] == "lastModifiedDateTime desc"
    assert "lastModifiedDateTime gt" in params["$filter"]
    assert kwargs["headers"] == _PREFER


def test_reader_duplicate_ids_last_occurrence_wins() -> None:
    http = MagicMock()
    first = _active_message_payload(body={"contentType": "text", "content": "first"})
    second = _active_message_payload(body={"contentType": "text", "content": "second"})
    http.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"value": [first, second]},
    )
    page = _reader(http).read_messages_snapshot_page(
        chat=_chat(),
        window=_window(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert page.items[0].body_content == "second"


def test_snapshot_page_is_complete_when_no_continuation() -> None:
    page = MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        window=_window(),
        items=(),
    )
    assert page.is_complete is True
