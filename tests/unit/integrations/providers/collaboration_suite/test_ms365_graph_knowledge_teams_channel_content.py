# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Teams Channel knowledge-read exact content surface."""

from __future__ import annotations

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
    ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    MsGraphKnowledgeTransport,
    MsGraphTeamsChatAttachmentKind,
    MsGraphTeamsChannelBodyKind,
    MsGraphTeamsChannelContentReader,
    MsGraphTeamsChannelContentTooLarge,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageReference,
    MsGraphTeamsChannelMessageType,
    MsGraphTeamsIdentityKind,
    validate_msgraph_teams_channel_message_content,
    validate_msgraph_teams_channel_message_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatMention,
    MsGraphTeamsChatReaction,
    MsGraphTeamsIdentity,
    validate_msgraph_teams_chat_attachment_reference,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_TEAM_ID = "team-abc-123"
_OTHER_TEAM_ID = "other-team-456"
_SPECIAL_TEAM_ID = "team/special"
_CHANNEL_ID = "channel-abc-123"
_OTHER_CHANNEL_ID = "other-channel-456"
_SPECIAL_CHANNEL_ID = "19:channel@thread.v2"
_MESSAGE_ID = "root-msg-001"
_ROOT_MESSAGE_ID = "root-msg-001"
_REPLY_MESSAGE_ID = "reply-msg-002"
_OTHER_ROOT_MESSAGE_ID = "other-root-999"
_OTHER_MESSAGE_ID = "other-msg-999"
_REPLY_ETAG = "reply-etag-2"
_SENDER_ID = "sender-secret-id"
_ETAG = "etag-1"
_OTHER_ETAG = "other-etag"
_SECRET_BODY = "secret-body-text-value"
_SECRET_SENDER_NAME = "Secret Sender Name"
_SECRET_ATTACHMENT_URL = "https://contoso.example/secret-file"
_SECRET_MENTION_TEXT = "secret-mention-text"
_SECRET_REACTION_TYPE = "secret-reaction"
_QUOTED_TEAM_ID = quote(_TEAM_ID, safe="")
_QUOTED_SPECIAL_TEAM_ID = quote(_SPECIAL_TEAM_ID, safe="")
_QUOTED_CHANNEL = quote(_CHANNEL_ID, safe="")
_QUOTED_SPECIAL_CHANNEL = quote(_SPECIAL_CHANNEL_ID, safe="")
_QUOTED_ROOT_MESSAGE_ID = quote(_ROOT_MESSAGE_ID, safe="")
_QUOTED_REPLY_MESSAGE_ID = quote(_REPLY_MESSAGE_ID, safe="")
_ROOT_CONTENT_PATH = (
    f"/teams/{_QUOTED_TEAM_ID}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}"
)
_REPLY_CONTENT_PATH = (
    f"/teams/{_QUOTED_TEAM_ID}/channels/{_QUOTED_CHANNEL}/messages/"
    f"{_QUOTED_ROOT_MESSAGE_ID}/replies/{_QUOTED_REPLY_MESSAGE_ID}"
)
_PREFER = {"Prefer": "include-unknown-enum-members"}
_SAFE_CONTENT = "unexpected Microsoft Graph Teams channel message content response"
_SAFE_MESSAGES = "unexpected Microsoft Graph Teams channel messages response"
_SAFE_KNOWLEDGE = "unexpected Microsoft Graph knowledge response"
_REQUEST_ERROR = "invalid Microsoft Graph Teams channel message content request"
_CHANGED_ERROR = "Microsoft Graph Teams channel message changed during read"
_TOO_LARGE_ERROR = "Microsoft Graph Teams channel message exceeds the configured content limit"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Teams channel message content capability"
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
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
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


def _active_reply_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": _REPLY_MESSAGE_ID,
        "replyToId": _ROOT_MESSAGE_ID,
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
        "etag": _REPLY_ETAG,
        "messageType": "message",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-01T11:00:00Z",
        "deletedDateTime": None,
        "importance": "normal",
        "body": {"contentType": "text", "content": "Reply body"},
        "from": {"user": {"id": "u2", "displayName": "Bob"}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base


def _valid_root_reference(**overrides: object) -> MsGraphTeamsChannelMessageReference:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessageReference(**defaults)


def _valid_reply_reference(**overrides: object) -> MsGraphTeamsChannelMessageReference:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _ROOT_MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.REPLY,
        "remote_id": _REPLY_MESSAGE_ID,
        "revision": _REPLY_ETAG,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessageReference(**defaults)


def _valid_active_message(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChannelBodyKind.TEXT,
        "body_content": "Hello",
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _valid_active_reply(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _ROOT_MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.REPLY,
        "remote_id": _REPLY_MESSAGE_ID,
        "revision": _REPLY_ETAG,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChannelBodyKind.TEXT,
        "body_content": "Reply body",
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _reader(http: MagicMock) -> MsGraphTeamsChannelContentReader:
    return MsGraphTeamsChannelContentReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _setup_root_happy_path(
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[MagicMock, MsGraphTeamsChannelContentReader]:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload or _active_message_payload())
    return http, _reader(http)


# --- constants ---


def test_default_max_chars_constants() -> None:
    assert DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS == 2_000_000
    assert ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS == 8_000_000


# --- reference validation ---


def test_validate_reference_accepts_mapping() -> None:
    validated = validate_msgraph_teams_channel_message_reference(
        {
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "thread_root_remote_id": _MESSAGE_ID,
            "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
            "remote_id": _MESSAGE_ID,
            "revision": _ETAG,
        }
    )
    assert validated.remote_id == _MESSAGE_ID


def test_reference_repr_hides_revision() -> None:
    reference = _valid_root_reference()
    rendered = repr(reference)
    assert _ETAG not in rendered
    assert _TEAM_ID in rendered


@pytest.mark.parametrize(
    "kwargs",
    [
        {"team_remote_id": None},
        {"channel_remote_id": None},
        {"thread_root_remote_id": None},
        {"remote_id": None},
        {"revision": None},
        {"message_kind": "root"},
        {"message_kind": "reply"},
        {"message_kind": "unknown"},
        {
            "thread_root_remote_id": _OTHER_MESSAGE_ID,
            "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        },
        {
            "thread_root_remote_id": _REPLY_MESSAGE_ID,
            "remote_id": _REPLY_MESSAGE_ID,
            "message_kind": MsGraphTeamsChannelMessageKind.REPLY,
        },
    ],
)
def test_model_construct_malformed_reference_rejected(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
    }
    malformed = MsGraphTeamsChannelMessageReference.model_construct(**{**base, **kwargs})
    with pytest.raises(ValueError, match=_SAFE_CONTENT) as exc:
        validate_msgraph_teams_channel_message_reference(malformed)
    assert exc.value.__cause__ is None
    assert _TEAM_ID not in str(exc.value)
    assert _ETAG not in str(exc.value)


# --- root success ---


def test_read_root_message_content_success() -> None:
    payload = _active_message_payload(
        body={"contentType": "text", "content": _SECRET_BODY},
        attachments=[
            {
                "id": "att-1",
                "contentType": "reference",
                "contentUrl": _SECRET_ATTACHMENT_URL,
            }
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
    http, reader = _setup_root_happy_path(payload=payload)
    result = reader.read_message_content(message=_valid_root_reference(), max_chars=10_000)
    assert result.body_content == _SECRET_BODY
    assert result.sender is not None
    assert result.sender.display_name == _SECRET_SENDER_NAME
    assert len(result.attachments) == 1
    assert result.attachments[0].content_url == _SECRET_ATTACHMENT_URL
    assert len(result.mentions) == 1
    assert result.mentions[0].mention_text == _SECRET_MENTION_TEXT
    assert len(result.reactions) == 1
    assert result.reactions[0].reaction_type == _SECRET_REACTION_TYPE
    assert result.remote_id == _MESSAGE_ID
    assert result.revision == _ETAG
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _ROOT_CONTENT_PATH
    assert http.get.call_args.kwargs["headers"] == _PREFER


def test_read_root_path_segments_are_url_quoted() -> None:
    special_payload = _active_message_payload(
        channelIdentity={"teamId": _SPECIAL_TEAM_ID, "channelId": _SPECIAL_CHANNEL_ID},
        id="msg/special",
    )
    http = MagicMock()
    http.get.return_value = _json_response(payload=special_payload)
    reference = _valid_root_reference(
        team_remote_id=_SPECIAL_TEAM_ID,
        channel_remote_id=_SPECIAL_CHANNEL_ID,
        thread_root_remote_id="msg/special",
        remote_id="msg/special",
    )
    _reader(http).read_message_content(message=reference, max_chars=10_000)
    expected_path = (
        f"/teams/{_QUOTED_SPECIAL_TEAM_ID}/channels/{_QUOTED_SPECIAL_CHANNEL}/messages/"
        f"{quote('msg/special', safe='')}"
    )
    assert http.get.call_args.args[0] == expected_path


def test_read_root_status_404_maps_to_dependency_error() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(status_code=404)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=10_000)


# --- reply success ---


def test_read_reply_message_content_success() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_reply_payload())
    result = _reader(http).read_message_content(message=_valid_reply_reference(), max_chars=10_000)
    assert result.message_kind is MsGraphTeamsChannelMessageKind.REPLY
    assert result.thread_root_remote_id == _ROOT_MESSAGE_ID
    assert result.remote_id == _REPLY_MESSAGE_ID
    assert result.revision == _REPLY_ETAG
    assert result.body_content == "Reply body"
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _REPLY_CONTENT_PATH
    assert "/replies/" in http.get.call_args.args[0]
    assert _QUOTED_ROOT_MESSAGE_ID in http.get.call_args.args[0]


# --- exact-reference mismatch ---


@pytest.mark.parametrize(
    ("message_overrides", "reference"),
    [
        ({"team_remote_id": _OTHER_TEAM_ID}, _valid_root_reference()),
        ({"channel_remote_id": _OTHER_CHANNEL_ID}, _valid_root_reference()),
        ({"thread_root_remote_id": _OTHER_ROOT_MESSAGE_ID}, _valid_reply_reference()),
        ({"remote_id": _OTHER_MESSAGE_ID}, _valid_root_reference()),
        ({"revision": _OTHER_ETAG}, _valid_root_reference()),
    ],
    ids=[
        "team_mismatch",
        "channel_mismatch",
        "thread_root_mismatch",
        "message_id_mismatch",
        "revision_mismatch",
    ],
)
def test_exact_reference_mismatch_raises_changed(
    message_overrides: dict[str, object],
    reference: MsGraphTeamsChannelMessageReference,
) -> None:
    if reference.message_kind is MsGraphTeamsChannelMessageKind.REPLY:
        message = _valid_active_reply(**message_overrides)
    elif "remote_id" in message_overrides:
        base = _valid_active_message()
        message = MsGraphTeamsChannelMessage.model_construct(
            team_remote_id=base.team_remote_id,
            channel_remote_id=base.channel_remote_id,
            thread_root_remote_id=base.thread_root_remote_id,
            message_kind=base.message_kind,
            remote_id=message_overrides["remote_id"],
            revision=base.revision,
            state=base.state,
            message_type=base.message_type,
            importance=base.importance,
            created_at=base.created_at,
            last_modified_at=base.last_modified_at,
            body_kind=base.body_kind,
            body_content=base.body_content,
        )
    else:
        message = _valid_active_message(**message_overrides)
    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR) as exc:
        validate_msgraph_teams_channel_message_content(
            message,
            reference=reference,
            max_chars=10_000,
        )
    rendered = str(exc.value)
    assert _SECRET_BODY not in rendered
    assert _TEAM_ID not in rendered
    assert _ETAG not in rendered
    assert _SECRET_ATTACHMENT_URL not in rendered


def test_root_reply_kind_mismatch_raises_changed() -> None:
    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR) as exc:
        validate_msgraph_teams_channel_message_content(
            _valid_active_reply(),
            reference=_valid_root_reference(),
            max_chars=10_000,
        )
    assert _REPLY_MESSAGE_ID not in str(exc.value)


def test_active_reference_deleted_message_raises_changed() -> None:
    deleted = _valid_active_message(
        state=MsGraphTeamsChannelMessageState.DELETED,
        deleted_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        body_kind=None,
        body_content=None,
    )
    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR) as exc:
        validate_msgraph_teams_channel_message_content(
            deleted,
            reference=_valid_root_reference(),
            max_chars=10_000,
        )
    assert _ETAG not in str(exc.value)


def test_reader_revision_mismatch_raises_changed() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_message_payload(etag=_OTHER_ETAG),
    )
    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=10_000)
    assert _OTHER_ETAG not in str(exc.value)


def test_reader_deleted_payload_raises_changed() -> None:
    http = MagicMock()
    deleted_payload = _active_message_payload(deletedDateTime="2024-01-01T12:00:00Z")
    deleted_payload.pop("body", None)
    deleted_payload.pop("from", None)
    deleted_payload.pop("attachments", None)
    deleted_payload.pop("mentions", None)
    deleted_payload.pop("reactions", None)
    http.get.return_value = _json_response(payload=deleted_payload)
    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR):
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=10_000)


# --- malformed provider response ---


def test_non_mapping_provider_payload_raises_safe_value_error() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=[])
    with pytest.raises(ValueError, match=_SAFE_KNOWLEDGE) as exc:
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=10_000)
    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        _active_message_payload(id=None),
        _active_message_payload(createdDateTime="not-a-date"),
        _active_message_payload(body={"contentType": "text"}),
        {**_active_message_payload(), "from": {"user": {"id": "u1", "displayName": 123}}},
        _active_message_payload(
            attachments=[{"id": "att", "contentType": "reference", "contentUrl": "http://bad"}]
        ),
        _active_message_payload(
            mentions=[{"id": 0, "mentionedText": "x", "mentioned": {"user": {}, "application": {}}}]
        ),
        _active_message_payload(
            reactions=[{"reactionType": "like", "createdDateTime": "bad", "user": {"user": {}}}]
        ),
    ],
)
def test_malformed_provider_payload_raises_safe_value_error(payload: object) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload)
    with pytest.raises(ValueError, match=_SAFE_MESSAGES) as exc:
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=10_000)
    assert exc.value.__cause__ is None
    assert _SECRET_BODY not in str(exc.value)


def test_model_construct_full_message_with_invalid_nested_fields_rejected() -> None:
    malformed = MsGraphTeamsChannelMessage.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChannelMessageState.ACTIVE,
        message_type=MsGraphTeamsChannelMessageType.MESSAGE,
        importance=MsGraphTeamsChannelImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChannelBodyKind.TEXT,
        body_content=123,
    )
    with pytest.raises(ValueError, match=_SAFE_CONTENT) as exc:
        validate_msgraph_teams_channel_message_content(
            malformed,
            reference=_valid_root_reference(),
            max_chars=10_000,
        )
    assert exc.value.__cause__ is None


# --- character limits ---


@pytest.mark.parametrize("max_chars", [DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS, ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS])
def test_valid_max_chars_accepted(max_chars: int) -> None:
    http, reader = _setup_root_happy_path()
    result = reader.read_message_content(message=_valid_root_reference(), max_chars=max_chars)
    assert result.body_content == "Hello"


def test_max_chars_one_accepted_for_single_character_body() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_message_payload(body={"contentType": "text", "content": "x"}),
    )
    result = _reader(http).read_message_content(message=_valid_root_reference(), max_chars=1)
    assert result.body_content == "x"


@pytest.mark.parametrize("max_chars", [0, -1, ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS + 1, True, False, "1000", 1.5, None])
def test_invalid_max_chars_rejected_before_http(max_chars: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_message_content(
            message=_valid_root_reference(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


def test_body_above_configured_max_raises_too_large() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_message_payload(body={"contentType": "text", "content": "a" * 600}),
    )
    with pytest.raises(MsGraphTeamsChannelContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=500)
    assert exc.value.__cause__ is None
    assert "aaa" not in str(exc.value)


def test_malformed_payload_does_not_raise_too_large() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_message_payload(body={"contentType": "text"}))
    with pytest.raises(ValueError, match=_SAFE_MESSAGES):
        _reader(http).read_message_content(message=_valid_root_reference(), max_chars=10_000)


# --- layer wiring ---


def test_graph_rest_client_delegates_content_read() -> None:
    http, _ = _setup_root_happy_path()
    result = _graph_client(http).read_teams_channel_message_content(message=_valid_root_reference())
    assert result.remote_id == _MESSAGE_ID


def test_collaboration_suite_delegates_content_read() -> None:
    http, _ = _setup_root_happy_path()
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    result = suite.read_teams_channel_message_content(message=_valid_root_reference())
    assert result.revision == _ETAG


def test_integration_delegates_content_read() -> None:
    http, _ = _setup_root_happy_path()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    result = integration.read_teams_channel_message_content(message=_valid_root_reference())
    assert result.body_content == "Hello"


def test_transport_and_reader_share_injected_http_client() -> None:
    http, _ = _setup_root_happy_path()
    client = _graph_client(http)
    client.read_teams_channel_message_content(message=_valid_root_reference())
    assert client._knowledge_transport._http_client is http
    assert client._teams_channel_content_reader._transport._http_client is http


class _CustomSuiteWithoutTeamsChannelContent(CollaborationSuite):
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


class _TrackingTeamsChannelContentClient(GraphRestClient):
    def __init__(self, message: MsGraphTeamsChannelMessage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_message = message
        self.last_reference: MsGraphTeamsChannelMessageReference | None = None
        self.last_max_chars: int | None = None

    def read_teams_channel_message_content(
        self,
        *,
        message: MsGraphTeamsChannelMessageReference,
        max_chars: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelMessage:
        self.last_reference = message
        self.last_max_chars = max_chars
        return self._custom_message


def test_custom_client_without_content_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutTeamsChannelContent(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        integration.read_teams_channel_message_content(message=_valid_root_reference())


def test_custom_client_forwards_validated_reference_and_max_chars() -> None:
    supplied = _valid_active_message(body_content="x" * 16_001)
    http = MagicMock()
    tracking = _TrackingTeamsChannelContentClient(message=supplied, http=http)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(tracking),
        enabled=True,
    )
    returned = integration.read_teams_channel_message_content(
        message=_valid_root_reference(),
        max_chars=20_000,
    )
    assert tracking.last_reference is not None
    assert tracking.last_reference.remote_id == _MESSAGE_ID
    assert tracking.last_reference.revision == _ETAG
    assert tracking.last_max_chars == 20_000
    assert returned.body_content == "x" * 16_001
    assert returned is not supplied
    http.get.assert_not_called()


def test_custom_client_malformed_content_rejected() -> None:
    malformed = MsGraphTeamsChannelMessage.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChannelMessageState.ACTIVE,
        message_type=MsGraphTeamsChannelMessageType.MESSAGE,
        importance=MsGraphTeamsChannelImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChannelBodyKind.TEXT,
        body_content=123,
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _TrackingTeamsChannelContentClient(message=malformed, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_CONTENT) as exc:
        integration.read_teams_channel_message_content(message=_valid_root_reference())
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("max_chars", [0, ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS + 1, True, "1000", None])
def test_integration_invalid_max_chars_rejected(max_chars: object) -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _TrackingTeamsChannelContentClient(message=_valid_active_message(), http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_teams_channel_message_content(
            message=_valid_root_reference(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )


def test_custom_client_identity_mismatch_rejected() -> None:
    base = _valid_active_message()
    supplied = MsGraphTeamsChannelMessage.model_construct(
        team_remote_id=base.team_remote_id,
        channel_remote_id=base.channel_remote_id,
        thread_root_remote_id=base.thread_root_remote_id,
        message_kind=base.message_kind,
        remote_id=_OTHER_MESSAGE_ID,
        revision=base.revision,
        state=base.state,
        message_type=base.message_type,
        importance=base.importance,
        created_at=base.created_at,
        last_modified_at=base.last_modified_at,
        body_kind=base.body_kind,
        body_content=base.body_content,
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _TrackingTeamsChannelContentClient(message=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR):
        integration.read_teams_channel_message_content(message=_valid_root_reference())


# --- repr and secret safety ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    attachment = MsGraphTeamsChatAttachmentReference(
        remote_id="att-1",
        attachment_kind=MsGraphTeamsChatAttachmentKind.REFERENCE,
        content_type="reference",
        content_url=_SECRET_ATTACHMENT_URL,
        has_thumbnail_url=False,
    )
    message = _valid_active_message(
        body_content=_SECRET_BODY,
        revision=_ETAG,
        sender=MsGraphTeamsIdentity(
            identity_kind=MsGraphTeamsIdentityKind.USER,
            remote_id=_SENDER_ID,
            display_name=_SECRET_SENDER_NAME,
        ),
        attachments=(attachment,),
        mentions=(
            MsGraphTeamsChatMention(
                mention_id=0,
                mention_text=_SECRET_MENTION_TEXT,
                mentioned=MsGraphTeamsIdentity(
                    identity_kind=MsGraphTeamsIdentityKind.USER,
                    remote_id="u3",
                    display_name="Carol",
                ),
            ),
        ),
        reactions=(
            MsGraphTeamsChatReaction(
                reaction_type=_SECRET_REACTION_TYPE,
                created_at=datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc),
                user=MsGraphTeamsIdentity(
                    identity_kind=MsGraphTeamsIdentityKind.USER,
                    remote_id="u4",
                    display_name="Dave",
                ),
            ),
        ),
    )
    reference = _valid_root_reference()
    rendered = repr(message)
    assert _SECRET_BODY not in rendered
    assert _SECRET_SENDER_NAME not in rendered
    assert _SECRET_ATTACHMENT_URL not in rendered
    assert _SECRET_MENTION_TEXT not in rendered
    assert _SECRET_REACTION_TYPE not in rendered
    assert _ETAG not in rendered

    reference_rendered = repr(reference)
    assert _ETAG not in reference_rendered

    validated_attachment = validate_msgraph_teams_chat_attachment_reference(attachment)
    assert _SECRET_ATTACHMENT_URL not in repr(validated_attachment)

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_message_content(
            message=_valid_root_reference(),
            max_chars=0,
        )
    assert _TEAM_ID not in str(exc.value)
    assert _ETAG not in str(exc.value)

    with pytest.raises(MsGraphTeamsChannelMessageChanged, match=_CHANGED_ERROR) as changed_exc:
        base = _valid_active_message()
        mismatch = MsGraphTeamsChannelMessage.model_construct(
            team_remote_id=base.team_remote_id,
            channel_remote_id=base.channel_remote_id,
            thread_root_remote_id=base.thread_root_remote_id,
            message_kind=base.message_kind,
            remote_id=_OTHER_MESSAGE_ID,
            revision=base.revision,
            state=base.state,
            message_type=base.message_type,
            importance=base.importance,
            created_at=base.created_at,
            last_modified_at=base.last_modified_at,
            body_kind=base.body_kind,
            body_content=base.body_content,
        )
        validate_msgraph_teams_channel_message_content(
            mismatch,
            reference=reference,
            max_chars=10_000,
        )
    changed_rendered = str(changed_exc.value)
    assert _OTHER_MESSAGE_ID not in changed_rendered
    assert _SECRET_BODY not in changed_rendered

    with pytest.raises(MsGraphTeamsChannelContentTooLarge, match=_TOO_LARGE_ERROR) as large_exc:
        validate_msgraph_teams_channel_message_content(
            _valid_active_message(body_content="a" * 600),
            reference=reference,
            max_chars=500,
        )
    assert "aaa" not in str(large_exc.value)
