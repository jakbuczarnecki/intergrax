# © Artur Czarnecki. All rights reserved.

"""Unit tests for Microsoft Graph Teams Chat knowledge-read exact content surface."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import (
    _Ms365GraphCollaborationSuite,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphKnowledgeTransport,
    MsGraphTeamsChatBodyKind,
    MsGraphTeamsChatContentReader,
    MsGraphTeamsChatContentTooLarge,
    MsGraphTeamsChatImportance,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageReference,
    MsGraphTeamsChatMessageType,
    validate_msgraph_teams_chat_message_content,
    validate_msgraph_teams_chat_message_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatMessageChanged,
    MsGraphTeamsChatMessageState,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX = "user@contoso.com"
_OTHER_MAILBOX = "other@contoso.com"
_CHAT_ID = "19:chat-abc@thread.v2"
_OTHER_CHAT_ID = "19:other-chat@thread.v2"
_SPECIAL_CHAT_ID = "19:chat/special@thread.v2"
_MESSAGE_ID = "msg-001"
_OTHER_MESSAGE_ID = "other-msg-999"
_SENDER_ID = "sender-secret-id"
_ETAG = "etag-1"
_OTHER_ETAG = "other-etag"
_SECRET_BODY = "secret-body-text-value"
_SECRET_SENDER_NAME = "Secret Sender Name"
_SECRET_ATTACHMENT_URL = "https://contoso.example/secret-file"
_SECRET_MENTION_TEXT = "secret-mention-text"
_SECRET_REACTION_TYPE = "secret-reaction"
_QUOTED_MAILBOX = quote(_MAILBOX, safe="")
_QUOTED_CHAT = quote(_CHAT_ID, safe="")
_QUOTED_SPECIAL_CHAT = quote(_SPECIAL_CHAT_ID, safe="")
_QUOTED_MESSAGE = quote(_MESSAGE_ID, safe="")
_CONTENT_PATH = f"/users/{_QUOTED_MAILBOX}/chats/{_QUOTED_CHAT}/messages/{_QUOTED_MESSAGE}"
_PREFER = {"Prefer": "include-unknown-enum-members"}
_SAFE_CONTENT = "unexpected Microsoft Graph Teams chat message content response"
_SAFE_MESSAGES = "unexpected Microsoft Graph Teams chat messages response"
_SAFE_KNOWLEDGE = "unexpected Microsoft Graph knowledge response"
_REQUEST_ERROR = "invalid Microsoft Graph Teams chat message content request"
_CHANGED_ERROR = "Microsoft Graph Teams chat message changed during read"
_TOO_LARGE_ERROR = "Microsoft Graph Teams chat message exceeds the configured content limit"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Teams chat message content capability"
)


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _json_response(*, status_code: int = 200, payload: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status = MagicMock()
    return response


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
        "from": {"user": {"id": _SENDER_ID, "displayName": _SECRET_SENDER_NAME}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base


def _valid_reference(**overrides: object) -> MsGraphTeamsChatMessageReference:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX,
        "chat_remote_id": _CHAT_ID,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
    }
    defaults.update(overrides)
    return MsGraphTeamsChatMessageReference(**defaults)


def _valid_active_message(**overrides: object) -> MsGraphTeamsChatMessage:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX,
        "chat_remote_id": _CHAT_ID,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
        "state": MsGraphTeamsChatMessageState.ACTIVE,
        "message_type": MsGraphTeamsChatMessageType.MESSAGE,
        "importance": MsGraphTeamsChatImportance.NORMAL,
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChatBodyKind.TEXT,
        "body_content": "Hello",
    }
    defaults.update(overrides)
    return MsGraphTeamsChatMessage(**defaults)


def _reader(http: MagicMock) -> MsGraphTeamsChatContentReader:
    return MsGraphTeamsChatContentReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _setup_happy_path(
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[MagicMock, MsGraphTeamsChatContentReader]:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload or _active_message_payload())
    return http, _reader(http)


def test_validate_reference_accepts_mapping() -> None:
    validated = validate_msgraph_teams_chat_message_reference(
        {
            "mailbox_user_id": _MAILBOX,
            "chat_remote_id": _CHAT_ID,
            "remote_id": _MESSAGE_ID,
            "revision": _ETAG,
        }
    )
    assert validated.remote_id == _MESSAGE_ID


def test_reference_repr_hides_identifiers() -> None:
    reference = _valid_reference()
    rendered = repr(reference)
    assert _ETAG not in rendered
    assert _MAILBOX not in rendered
    assert _CHAT_ID not in rendered
    assert _MESSAGE_ID not in rendered


def test_read_active_text_message() -> None:
    http, reader = _setup_happy_path()
    result = reader.read_message_content(message=_valid_reference(), max_chars=10_000)
    assert result.body_content == "Hello"
    assert result.body_kind is MsGraphTeamsChatBodyKind.TEXT
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _CONTENT_PATH
    assert http.get.call_args.kwargs["headers"] == _PREFER


def test_read_active_html_message() -> None:
    payload = _active_message_payload(
        body={"contentType": "html", "content": "<p>Hi</p>"},
    )
    http, reader = _setup_happy_path(payload=payload)
    result = reader.read_message_content(message=_valid_reference(), max_chars=10_000)
    assert result.body_kind is MsGraphTeamsChatBodyKind.HTML
    assert result.body_content == "<p>Hi</p>"


def test_read_preserves_sender_mentions_reactions_attachments_forwarded() -> None:
    forwarded_content = json.dumps(
        {
            "originalMessageId": "orig-msg",
            "originalConversationId": _CHAT_ID,
            "originalSentDateTime": "2024-01-01T09:00:00Z",
        }
    )
    payload = _active_message_payload(
        body={"contentType": "text", "content": _SECRET_BODY},
        attachments=[
            {
                "id": "att-1",
                "contentType": "reference",
                "contentUrl": _SECRET_ATTACHMENT_URL,
            },
            {
                "id": "att-fwd",
                "contentType": "forwardedMessageReference",
                "content": forwarded_content,
            },
        ],
        mentions=[
            {
                "id": 0,
                "mentionedText": _SECRET_MENTION_TEXT,
                "mentioned": {"user": {"id": "u3", "displayName": "Carol"}},
            }
        ],
        reactions=[
            {
                "reactionType": _SECRET_REACTION_TYPE,
                "createdDateTime": "2024-01-01T10:30:00Z",
                "user": {"user": {"id": "u4", "displayName": "Dave"}},
            }
        ],
    )
    http, reader = _setup_happy_path(payload=payload)
    result = reader.read_message_content(message=_valid_reference(), max_chars=10_000)
    assert result.sender is not None
    assert result.sender.display_name == _SECRET_SENDER_NAME
    assert len(result.attachments) == 2
    assert result.attachments[0].content_url == _SECRET_ATTACHMENT_URL
    assert result.attachments[1].forwarded_message is not None
    assert len(result.mentions) == 1
    assert result.mentions[0].mention_text == _SECRET_MENTION_TEXT
    assert len(result.reactions) == 1
    assert result.reactions[0].reaction_type == _SECRET_REACTION_TYPE


def test_read_path_segments_are_url_quoted() -> None:
    special_payload = _active_message_payload(
        chatId=_SPECIAL_CHAT_ID,
        id="msg/special",
    )
    http = MagicMock()
    http.get.return_value = _json_response(payload=special_payload)
    reference = _valid_reference(
        chat_remote_id=_SPECIAL_CHAT_ID,
        remote_id="msg/special",
    )
    _reader(http).read_message_content(message=reference, max_chars=10_000)
    expected_path = (
        f"/users/{_QUOTED_MAILBOX}/chats/{_QUOTED_SPECIAL_CHAT}/messages/"
        f"{quote('msg/special', safe='')}"
    )
    assert http.get.call_args.args[0] == expected_path


@pytest.mark.parametrize(
    ("message_overrides", "reference_overrides"),
    [
        ({"revision": _OTHER_ETAG}, {}),
        ({"remote_id": _OTHER_MESSAGE_ID}, {}),
        ({"chat_remote_id": _OTHER_CHAT_ID}, {}),
        ({}, {"mailbox_user_id": _OTHER_MAILBOX}),
    ],
    ids=["revision", "message_id", "chat_id", "mailbox"],
)
def test_exact_reference_mismatch_raises_changed(
    message_overrides: dict[str, object],
    reference_overrides: dict[str, object],
) -> None:
    message = _valid_active_message(**message_overrides)
    reference = _valid_reference(**reference_overrides)
    with pytest.raises(MsGraphTeamsChatMessageChanged, match=_CHANGED_ERROR) as exc:
        validate_msgraph_teams_chat_message_content(
            message,
            reference=reference,
            max_chars=10_000,
        )
    rendered = str(exc.value)
    assert _SECRET_BODY not in rendered
    assert _ETAG not in rendered


def test_deleted_message_raises_changed() -> None:
    deleted = _valid_active_message(
        state=MsGraphTeamsChatMessageState.DELETED,
        deleted_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        body_kind=None,
        body_content=None,
    )
    with pytest.raises(MsGraphTeamsChatMessageChanged, match=_CHANGED_ERROR):
        validate_msgraph_teams_chat_message_content(
            deleted,
            reference=_valid_reference(),
            max_chars=10_000,
        )


def test_reader_revision_mismatch_raises_changed() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_message_payload(etag=_OTHER_ETAG))
    with pytest.raises(MsGraphTeamsChatMessageChanged, match=_CHANGED_ERROR):
        _reader(http).read_message_content(message=_valid_reference(), max_chars=10_000)


def test_reader_deleted_payload_raises_changed() -> None:
    http = MagicMock()
    deleted_payload = _active_message_payload(deletedDateTime="2024-01-01T12:00:00Z")
    for key in ("body", "from", "attachments", "mentions", "reactions"):
        deleted_payload.pop(key, None)
    http.get.return_value = _json_response(payload=deleted_payload)
    with pytest.raises(MsGraphTeamsChatMessageChanged, match=_CHANGED_ERROR):
        _reader(http).read_message_content(message=_valid_reference(), max_chars=10_000)


def test_reader_rejects_channel_identity() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_message_payload(channelIdentity={"teamId": "t"}),
    )
    with pytest.raises(ValueError, match=_SAFE_MESSAGES):
        _reader(http).read_message_content(message=_valid_reference(), max_chars=10_000)


def test_reader_rejects_reply_to_id() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_message_payload(replyToId="parent"))
    with pytest.raises(ValueError, match=_SAFE_MESSAGES):
        _reader(http).read_message_content(message=_valid_reference(), max_chars=10_000)


def test_provider_404_maps_to_dependency_error() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(status_code=404)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        _reader(http).read_message_content(message=_valid_reference(), max_chars=10_000)


def test_invalid_reference_rejected_before_http() -> None:
    http = MagicMock()
    bad = MsGraphTeamsChatMessageReference.model_construct(
        mailbox_user_id="",
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
    )
    with pytest.raises(ValueError, match=_SAFE_CONTENT):
        _reader(http).read_message_content(message=bad, max_chars=10_000)
    http.get.assert_not_called()


@pytest.mark.parametrize("max_chars", [0, -1, ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS + 1, "100", None])
def test_invalid_max_chars_rejected_before_http(max_chars: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_message_content(
            message=_valid_reference(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


def test_body_above_configured_max_raises_too_large() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_message_payload(body={"contentType": "text", "content": "a" * 600}),
    )
    with pytest.raises(MsGraphTeamsChatContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_reference(), max_chars=500)
    assert exc.value.__cause__ is None
    assert "aaa" not in str(exc.value)


def test_malformed_provider_payload_raises_safe_error() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_message_payload(body={"contentType": "text"}))
    with pytest.raises(ValueError, match=_SAFE_MESSAGES):
        _reader(http).read_message_content(message=_valid_reference(), max_chars=10_000)


def test_graph_rest_client_delegates_content_read() -> None:
    http, _ = _setup_happy_path()
    result = _graph_client(http).read_teams_chat_message_content(message=_valid_reference())
    assert result.remote_id == _MESSAGE_ID


def test_collaboration_suite_delegates_content_read() -> None:
    http, _ = _setup_happy_path()
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    result = suite.read_teams_chat_message_content(message=_valid_reference())
    assert result.revision == _ETAG


def test_integration_delegates_content_read() -> None:
    http, _ = _setup_happy_path()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    result = integration.read_teams_chat_message_content(message=_valid_reference())
    assert result.body_content == "Hello"


def test_transport_and_reader_share_injected_http_client() -> None:
    http, _ = _setup_happy_path()
    client = _graph_client(http)
    client.read_teams_chat_message_content(message=_valid_reference())
    assert client._knowledge_transport._http_client is http
    assert client._teams_chat_content_reader._transport._http_client is http


class _CustomSuiteWithoutTeamsChatContent(CollaborationSuite):
    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(self, user_id: str, *, start: str, end: str, limit: int = 50):
        raise NotImplementedError

    def get_user(self, user_id: str):
        raise NotImplementedError

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        raise NotImplementedError

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees=(),
    ):
        raise NotImplementedError


def test_integration_missing_content_capability_raises_configuration_error() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutTeamsChatContent(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        integration.read_teams_chat_message_content(message=_valid_reference())


def test_security_exception_and_reference_repr_hide_secrets() -> None:
    reference = _valid_reference()
    assert _ETAG not in repr(reference)
    with pytest.raises(MsGraphTeamsChatContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        validate_msgraph_teams_chat_message_content(
            _valid_active_message(body_content="x" * 20),
            reference=reference,
            max_chars=5,
        )
    assert "xxxx" not in str(exc.value)
