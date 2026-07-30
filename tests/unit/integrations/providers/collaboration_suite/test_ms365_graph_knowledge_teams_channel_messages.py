# © Artur Czarnecki. All rights reserved.

"""Unit tests for Microsoft Graph Teams Channel knowledge-read messages surface."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationConfigurationError
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
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    MsGraphTeamsChatAttachmentKind,
    MsGraphTeamsChannelBodyKind,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageType,
    MsGraphTeamsIdentityKind,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MsGraphTeamsChannel,
    parse_msgraph_teams_channel,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessagesReader,
    MsGraphTeamsChannelReplyPage,
    MsGraphTeamsChannelRootMessagePage,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChannelRootMessageReference,
    parse_msgraph_teams_channel_message,
    read_and_validate_current_teams_channel_message_observation,
    root_message_reference_from_message,
    validate_msgraph_teams_channel_message,
    validate_msgraph_teams_channel_reply_page,
    validate_msgraph_teams_channel_replies_continuation,
    validate_msgraph_teams_channel_root_message_reference,
    validate_msgraph_teams_channel_root_messages_continuation,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsForwardedMessageReference,
    MsGraphTeamsIdentity,
    validate_msgraph_teams_chat_attachment_reference,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_TEAM_ID = "team-abc-123"
_OTHER_TEAM_ID = "other-team-456"
_CHANNEL_ID = "channel-abc-123"
_OTHER_CHANNEL_ID = "other-channel-456"
_MESSAGE_ID = "root-msg-001"
_ROOT_MESSAGE_ID = "root-msg-001"
_REPLY_MESSAGE_ID = "reply-msg-002"
_OTHER_ROOT_MESSAGE_ID = "other-root-999"
_REPLY_ETAG = "reply-etag-2"
_SENDER_ID = "sender-secret-id"
_ETAG = "etag-1"
_SECRET_TOKEN = "secret-skiptoken-value"
_QUOTED_TEAM_ID = quote(_TEAM_ID, safe="")
_QUOTED_OTHER_TEAM_ID = quote(_OTHER_TEAM_ID, safe="")
_QUOTED_CHANNEL = quote(_CHANNEL_ID, safe="")
_QUOTED_ROOT_MESSAGE_ID = quote(_ROOT_MESSAGE_ID, safe="")
_QUOTED_REPLY_MESSAGE_ID = quote(_REPLY_MESSAGE_ID, safe="")
_ROOT_OBSERVATION_PATH = (
    f"/teams/{_QUOTED_TEAM_ID}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}"
)
_REPLIES_COLLECTION_PATH = f"{_ROOT_OBSERVATION_PATH}/replies"
_REPLY_OBSERVATION_PATH = f"{_REPLIES_COLLECTION_PATH}/{_QUOTED_REPLY_MESSAGE_ID}"
_PREFER = {"Prefer": "include-unknown-enum-members"}
_SAFE = "unexpected Microsoft Graph Teams channel messages response"
_CHAT_PRIMITIVE_SAFE = "unexpected Microsoft Graph Teams chat messages response"
_CONT = "invalid Microsoft Graph Teams channel messages continuation"
_REQUEST_ERROR = "invalid Microsoft Graph Teams channel messages request"
_VALIDATION_ERROR = "Microsoft Graph Teams Channel validation is not configured"


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _channel() -> MsGraphTeamsChannel:
    return parse_msgraph_teams_channel(
        {
            "id": _CHANNEL_ID,
            "displayName": "General",
            "membershipType": "standard",
            "isArchived": False,
            "createdDateTime": "2024-01-01T10:00:00Z",
        },
        expected_team_id=_TEAM_ID,
    )


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
        "from": {"user": {"id": "u1", "displayName": "Alice"}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base



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


def _active_reply_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
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


def _valid_deleted_reply(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _ROOT_MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.REPLY,
        "remote_id": _REPLY_MESSAGE_ID,
        "revision": _REPLY_ETAG,
        "state": MsGraphTeamsChannelMessageState.DELETED,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        "deleted_at": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _json_response(payload: object) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def _replies_next_link(
    root_id: str = _ROOT_MESSAGE_ID,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    quoted_root = quote(root_id, safe="")
    quoted_channel = quote(channel_id, safe="")
    quoted_team = quote(team_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams/{quoted_team}/channels/"
        f"{quoted_channel}/messages/{quoted_root}/replies?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_replies_next_link(
    root_id: str = _ROOT_MESSAGE_ID,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    escaped_root = root_id.replace("'", "''")
    escaped_channel = channel_id.replace("'", "''")
    escaped_team = team_id.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/teams('{escaped_team}')/channels('{escaped_channel}')"
        f"/messages('{escaped_root}')/replies?$skiptoken={_SECRET_TOKEN}"
    )


def _slash_replies_next_link(root_id: str, channel_id: str = _CHANNEL_ID) -> str:
    quoted_root = quote(root_id, safe="")
    quoted_channel = quote(channel_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{quoted_channel}/messages/{quoted_root}/replies?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_percent_encoded_replies_next_link(
    root_id: str,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    escaped_root = root_id.replace("'", "''")
    escaped_channel = channel_id.replace("'", "''")
    escaped_team = team_id.replace("'", "''")
    encoded_root = quote(escaped_root, safe="")
    encoded_channel = quote(escaped_channel, safe="")
    encoded_team = quote(escaped_team, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams('{encoded_team}')/channels('{encoded_channel}')"
        f"/messages('{encoded_root}')/replies?$skiptoken={_SECRET_TOKEN}"
    )


def _valid_reply_page(
    *,
    continuation: MsGraphKnowledgeContinuation | None = None,
) -> MsGraphTeamsChannelReplyPage:
    return MsGraphTeamsChannelReplyPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        root_message_remote_id=_ROOT_MESSAGE_ID,
        root_message_revision=_ETAG,
        items=(_valid_active_reply(),),
        continuation=continuation,
    )


def _setup_replies_page_http(
    http: MagicMock,
    *,
    reply_items: list[dict[str, object]] | None = None,
    next_link: str | None = None,
) -> None:
    root_observation = _active_message_payload()
    replies_payload: dict[str, object] = {
        "value": reply_items if reply_items is not None else [_active_reply_payload()],
    }
    if next_link is not None:
        replies_payload["@odata.nextLink"] = next_link
    http.get.side_effect = [
        _json_response(payload=root_observation),
        _json_response(payload=replies_payload),
        _json_response(payload=root_observation),
    ]


def _reference_attachment(**overrides: object) -> MsGraphTeamsChatAttachmentReference:
    defaults: dict[str, object] = {
        "remote_id": "att-1",
        "attachment_kind": MsGraphTeamsChatAttachmentKind.REFERENCE,
        "content_type": "reference",
        "content_url": "https://contoso.sharepoint.com/file",
        "has_thumbnail_url": False,
    }
    defaults.update(overrides)
    return MsGraphTeamsChatAttachmentReference(**defaults)




def test_parse_active_text_message() -> None:
    msg = parse_msgraph_teams_channel_message(
        _active_message_payload(),
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.state is MsGraphTeamsChannelMessageState.ACTIVE
    assert msg.body_content == "Hello"
    assert msg.sender is not None
    assert msg.sender.display_name == "Alice"


def test_parse_deleted_message() -> None:
    msg = parse_msgraph_teams_channel_message(
        {
            "id": _MESSAGE_ID,
            "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
            "etag": _ETAG,
            "messageType": "message",
            "createdDateTime": "2024-01-01T10:00:00Z",
            "lastModifiedDateTime": "2024-01-01T12:00:00Z",
            "deletedDateTime": "2024-01-01T12:00:00Z",
            "importance": "normal",
        },
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.state is MsGraphTeamsChannelMessageState.DELETED
    assert msg.deleted_at is not None
    assert msg.body_kind is None


def test_parse_reference_attachment_https_only() -> None:
    msg = parse_msgraph_teams_channel_message(
        _active_message_payload(
            attachments=[
                {
                    "id": "att-1",
                    "contentType": "reference",
                    "contentUrl": "https://contoso.sharepoint.com/file",
                }
            ]
        ),
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.attachments[0].content_url is not None
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_message_payload(
                attachments=[
                    {
                        "id": "att-1",
                        "contentType": "reference",
                        "contentUrl": "http://insecure.example/file",
                    }
                ]
            ),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_channel_identity_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_message_payload(channelIdentity={"teamId": "t"}),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_user_identity_type_from_user_identity_type_field() -> None:
    payload = _active_message_payload()
    payload["from"] = {
        "user": {
            "id": _SENDER_ID,
            "displayName": "Alice",
            "userIdentityType": "aadUser",
            "@odata.type": "#microsoft.graph.teamworkUserIdentity",
        }
    }
    msg = parse_msgraph_teams_channel_message(
        payload,
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.sender is not None
    assert msg.sender.identity_type == "aadUser"
    assert msg.sender.identity_type != "#microsoft.graph.teamworkUserIdentity"


def test_parse_application_identity_type() -> None:
    payload = _active_message_payload()
    payload["from"] = {
        "application": {
            "id": "app-1",
            "displayName": "Bot",
            "applicationIdentityType": "bot",
            "@odata.type": "#microsoft.graph.teamworkApplicationIdentity",
        }
    }
    msg = parse_msgraph_teams_channel_message(
        payload,
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.sender is not None
    assert msg.sender.identity_type == "bot"


def test_parse_unknown_identity_type_retained() -> None:
    payload = _active_message_payload()
    payload["from"] = {
        "user": {
            "id": _SENDER_ID,
            "displayName": "Alice",
            "userIdentityType": "futureProviderValue",
        }
    }
    msg = parse_msgraph_teams_channel_message(
        payload,
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.sender is not None
    assert msg.sender.identity_type == "futureProviderValue"


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS + 1, True, False, "100", 1.5, None],
)
def test_validate_message_rejects_invalid_max_chars(max_chars: object) -> None:
    message = _valid_active_message()
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message, max_chars=max_chars)  # type: ignore[arg-type]


def test_validate_message_body_at_max_chars_succeeds() -> None:
    body = "x" * 100
    message = _valid_active_message(body_content=body)
    validated = validate_msgraph_teams_channel_message(message, max_chars=100)
    assert validated.body_content == body
    assert validated is not message


def test_validate_message_body_one_over_max_chars_fails() -> None:
    message = _valid_active_message(body_content="x" * 101)
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message, max_chars=100)


def test_validate_message_body_with_nul_fails() -> None:
    message = MsGraphTeamsChannelMessage.model_construct(
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
        body_content="a\x00b",
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message, max_chars=100)


def test_validate_message_missing_body_content_rejected() -> None:
    message = MsGraphTeamsChannelMessage.model_construct(
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
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message)


def test_validate_message_integer_body_content_rejected() -> None:
    message = MsGraphTeamsChannelMessage.model_construct(
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
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message)


def test_validate_message_malformed_sender_rejected() -> None:
    message = MsGraphTeamsChannelMessage.model_construct(
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
        body_content="hello",
        sender=MsGraphTeamsIdentity.model_construct(),
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message)


def test_validate_message_malformed_attachment_rejected() -> None:
    message = MsGraphTeamsChannelMessage.model_construct(
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
        body_content="hello",
        attachments=(
            MsGraphTeamsChatAttachmentReference.model_construct(
                attachment_kind="reference",
                content_type="reference",
            ),
        ),
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_channel_message(message)


def test_parse_body_exactly_at_max_chars() -> None:
    body = "a" * 50
    msg = parse_msgraph_teams_channel_message(
        _active_message_payload(body={"contentType": "text", "content": body}),
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=50,
    )
    assert msg.body_content == body


def test_parse_body_one_over_max_chars_fails() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_message_payload(body={"contentType": "text", "content": "a" * 51}),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            max_chars=50,
        )


def test_parse_body_with_nul_fails() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_message_payload(body={"contentType": "text", "content": "a\x00b"}),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            max_chars=100,
        )



def test_attachment_reference_model_construct_raw_kind_rejected() -> None:
    malformed = MsGraphTeamsChatAttachmentReference.model_construct(
        remote_id="att-raw-kind",
        attachment_kind="reference",
        content_type="reference",
        content_url="https://contoso.example/file",
        embedded_content=None,
        forwarded_message=None,
        has_thumbnail_url=False,
    )
    with pytest.raises(
        ValueError,
        match=_CHAT_PRIMITIVE_SAFE,
    ) as exc:
        validate_msgraph_teams_chat_attachment_reference(malformed)
    assert exc.value.__cause__ is None
    assert "https://contoso.example/file" not in str(exc.value)
    assert "att-raw-kind" not in str(exc.value)


@pytest.mark.parametrize(
    ("raw_kind", "attachment_kwargs"),
    [
        (
            "reference",
            {
                "remote_id": "att-ref",
                "content_type": "reference",
                "content_url": "https://contoso.example/file",
                "embedded_content": None,
                "forwarded_message": None,
                "has_thumbnail_url": False,
            },
        ),
        (
            "forwarded_message_reference",
            {
                "remote_id": "att-fwd",
                "content_type": "forwardedMessageReference",
                "content_url": None,
                "embedded_content": None,
                "forwarded_message": MsGraphTeamsForwardedMessageReference(
                    original_message_id="orig-msg",
                    original_chat_id=_CHANNEL_ID,
                    original_sent_at=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                    original_sender=None,
                ),
                "has_thumbnail_url": False,
            },
        ),
        (
            "code_snippet",
            {
                "remote_id": "att-code",
                "content_type": "application/vnd.microsoft.card.codesnippet",
                "content_url": None,
                "embedded_content": '{"language":"python"}',
                "forwarded_message": None,
                "has_thumbnail_url": False,
            },
        ),
        (
            "announcement",
            {
                "remote_id": "att-announce",
                "content_type": "application/vnd.microsoft.card.announcement",
                "content_url": None,
                "embedded_content": '{"title":"Update"}',
                "forwarded_message": None,
                "has_thumbnail_url": False,
            },
        ),
        (
            "card",
            {
                "remote_id": "att-card",
                "content_type": "application/vnd.microsoft.card.adaptive",
                "content_url": None,
                "embedded_content": '{"type":"AdaptiveCard"}',
                "forwarded_message": None,
                "has_thumbnail_url": False,
            },
        ),
        (
            "unknown",
            {
                "remote_id": "att-unknown",
                "content_type": "application/octet-stream",
                "content_url": None,
                "embedded_content": "opaque",
                "forwarded_message": None,
                "has_thumbnail_url": False,
            },
        ),
    ],
)
def test_attachment_reference_raw_kind_string_rejected(
    raw_kind: str,
    attachment_kwargs: dict[str, object],
) -> None:
    malformed = MsGraphTeamsChatAttachmentReference.model_construct(
        attachment_kind=raw_kind,
        **attachment_kwargs,
    )
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE) as exc:
        validate_msgraph_teams_chat_attachment_reference(malformed)
    assert exc.value.__cause__ is None


def test_attachment_reference_valid_enum_instance_accepted() -> None:
    attachment = MsGraphTeamsChatAttachmentReference.model_construct(
        remote_id="att-valid-kind",
        attachment_kind=MsGraphTeamsChatAttachmentKind.REFERENCE,
        content_type="reference",
        content_url="https://contoso.example/file",
        embedded_content=None,
        forwarded_message=None,
        has_thumbnail_url=False,
    )
    validated = validate_msgraph_teams_chat_attachment_reference(attachment)
    assert validated is not attachment
    assert validated.content_url == "https://contoso.example/file"


def test_attachment_reference_valid_https() -> None:
    attachment = _reference_attachment()
    validated = validate_msgraph_teams_chat_attachment_reference(attachment)
    assert validated.content_url is not None
    assert validated is not attachment


def test_attachment_reference_missing_url_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(content_url=None)
        )


def test_attachment_reference_http_url_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(content_url="http://insecure.example/file")
        )


def test_attachment_reference_userinfo_url_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(content_url="https://user:pass@contoso.example/file")
        )


def test_attachment_reference_with_embedded_content_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(embedded_content="inline")
        )


def test_attachment_forwarded_valid() -> None:
    forwarded = MsGraphTeamsForwardedMessageReference(
        original_message_id="orig-msg",
        original_chat_id=_CHANNEL_ID,
        original_sent_at=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
        original_sender=MsGraphTeamsIdentity(
            identity_kind=MsGraphTeamsIdentityKind.USER,
            remote_id=_SENDER_ID,
            display_name="Alice",
        ),
    )
    attachment = MsGraphTeamsChatAttachmentReference(
        remote_id="att-fwd",
        attachment_kind=MsGraphTeamsChatAttachmentKind.FORWARDED_MESSAGE_REFERENCE,
        content_type="forwardedMessageReference",
        forwarded_message=forwarded,
        has_thumbnail_url=False,
    )
    validated = validate_msgraph_teams_chat_attachment_reference(attachment)
    assert validated.forwarded_message is not None
    assert validated.forwarded_message.original_sender is not None
    assert validated.forwarded_message is not forwarded


def test_attachment_forwarded_missing_metadata_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            MsGraphTeamsChatAttachmentReference(
                remote_id="att-fwd",
                attachment_kind=MsGraphTeamsChatAttachmentKind.FORWARDED_MESSAGE_REFERENCE,
                content_type="forwardedMessageReference",
                has_thumbnail_url=False,
            )
        )


def test_attachment_kind_inconsistent_with_content_type_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(
                attachment_kind=MsGraphTeamsChatAttachmentKind.CARD,
            )
        )


def test_attachment_card_with_content_url_rejected() -> None:
    with pytest.raises(ValueError, match=_CHAT_PRIMITIVE_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            MsGraphTeamsChatAttachmentReference(
                remote_id="att-card",
                attachment_kind=MsGraphTeamsChatAttachmentKind.CARD,
                content_type="application/vnd.microsoft.card.adaptive",
                content_url="https://contoso.example/card",
                has_thumbnail_url=False,
            )
        )


def test_messages_continuation_validation() -> None:
    url = (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages?$skiptoken=x"
    )
    cont = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    validate_msgraph_teams_channel_root_messages_continuation(
        cont, team_id=_TEAM_ID, channel_id=_CHANNEL_ID, graph_base_url=_GRAPH_BASE
    )


def _reader(http: MagicMock) -> MsGraphTeamsChannelMessagesReader:
    return MsGraphTeamsChannelMessagesReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def test_reader_root_request_params() -> None:
    http = MagicMock()
    http.get.return_value = MagicMock(status_code=200, json=lambda: {"value": []})
    _reader(http).read_teams_channel_root_messages_page(
        channel=_channel(),
        continuation=None,
        limit=25,
        max_chars_per_message=1000,
    )
    _, kwargs = http.get.call_args
    params = kwargs["params"]
    assert params["$top"] == 25
    assert "$filter" not in params
    assert "$orderby" not in params
    assert "$select" not in params
    assert kwargs["headers"] == _PREFER


def test_reader_duplicate_ids_last_occurrence_wins() -> None:
    http = MagicMock()
    first = _active_message_payload(body={"contentType": "text", "content": "first"})
    second = _active_message_payload(body={"contentType": "text", "content": "second"})
    http.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"value": [first, second]},
    )
    page = _reader(http).read_teams_channel_root_messages_page(
        channel=_channel(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert page.items[0].body_content == "second"


def test_root_page_is_complete_when_no_continuation() -> None:
    page = MsGraphTeamsChannelRootMessagePage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        items=(),
    )
    assert page.is_complete is True


def test_invalid_continuation_rejected_before_messages_request() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM_ID}/channels/"
            f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT):
        _reader(http).read_teams_channel_root_messages_page(
            channel=_channel(),
            continuation=continuation,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    http.get.assert_not_called()


class _CountingMessagesClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChannelRootMessagePage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page
        self.call_count = 0
        self.last_continuation: MsGraphKnowledgeContinuation | None = None

    def read_teams_channel_root_messages_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelRootMessagePage:
        self.call_count += 1
        self.last_continuation = continuation
        return self._custom_page


class _CustomMessagesSuite(CollaborationSuite):
    def __init__(self, client: _CountingMessagesClient) -> None:
        self._client = client

    def read_teams_channel_root_messages_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelRootMessagePage:
        return self._client.read_teams_channel_root_messages_page(
            channel=channel,
            continuation=continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )

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


def _valid_root_page() -> MsGraphTeamsChannelRootMessagePage:
    return MsGraphTeamsChannelRootMessagePage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        items=(_valid_active_message(),),
    )


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
                f"{_QUOTED_CHANNEL}/messages/delta?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM_ID}/channels/"
                f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
                f"{quote(_OTHER_CHANNEL_ID, safe='')}/messages?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
    ],
)
def test_integration_rejects_malformed_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    client = _CountingMessagesClient(page=_valid_root_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT) as exc:
        integration.read_teams_channel_root_messages_page(
            channel=_channel(),
            continuation=continuation,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS + 1, True, False, "100", 1.5, None],
)
def test_integration_rejects_invalid_max_chars_before_custom_call(max_chars: object) -> None:
    client = _CountingMessagesClient(page=_valid_root_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_teams_channel_root_messages_page(
            channel=_channel(),
            continuation=None,
            limit=50,
            max_chars_per_message=max_chars,  # type: ignore[arg-type]
        )
    assert client.call_count == 0


def test_integration_valid_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
            f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    client = _CountingMessagesClient(page=_valid_root_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_channel_root_messages_page(
        channel=_channel(),
        continuation=continuation,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not _valid_root_page().items[0]


def test_integration_custom_client_raw_attachment_kind_rejected() -> None:
    message = MsGraphTeamsChannelMessage.model_construct(
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
        body_content="Hello",
        attachments=(
            MsGraphTeamsChatAttachmentReference.model_construct(
                remote_id="att-raw-kind",
                attachment_kind="reference",
                content_type="reference",
                content_url="https://contoso.example/file",
                embedded_content=None,
                forwarded_message=None,
                has_thumbnail_url=False,
            ),
        ),
    )
    page = MsGraphTeamsChannelRootMessagePage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        items=(message,),
    )
    client = _CountingMessagesClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE) as exc:
        integration.read_teams_channel_root_messages_page(
            channel=_channel(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    assert client.call_count == 1
    assert exc.value.__cause__ is None
    error_text = str(exc.value)
    assert "Hello" not in error_text
    assert "https://contoso.example/file" not in error_text
    assert "att-raw-kind" not in error_text
    assert _CHANNEL_ID not in error_text


def test_integration_custom_client_body_above_requested_limit_rejected() -> None:
    oversized = _valid_active_message(body_content="x" * 200)
    page = MsGraphTeamsChannelRootMessagePage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        items=(oversized,),
    )
    client = _CountingMessagesClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE):
        integration.read_teams_channel_root_messages_page(
            channel=_channel(),
            continuation=None,
            limit=50,
            max_chars_per_message=100,
        )


def test_security_repr_hides_sensitive_identifiers() -> None:
    identity = MsGraphTeamsIdentity(
        identity_kind=MsGraphTeamsIdentityKind.USER,
        remote_id=_SENDER_ID,
        display_name="Alice",
    )
    assert _SENDER_ID not in repr(identity)

    message = _valid_active_message(
        sender=identity,
        attachments=(_reference_attachment(),),
    )
    rendered = repr(message)
    assert _SENDER_ID not in rendered
    assert "Hello" not in rendered
    assert _TEAM_ID not in rendered
    assert _CHANNEL_ID not in rendered
    assert _MESSAGE_ID not in rendered
    assert _ETAG not in rendered
    assert message.team_remote_id == _TEAM_ID
    assert message.remote_id == _MESSAGE_ID


def test_parse_forwarded_message_reference_from_provider_payload() -> None:
    forwarded_content = json.dumps(
        {
            "originalMessageId": "orig-msg",
            "originalConversationId": _CHANNEL_ID,
            "originalSentDateTime": "2024-01-01T09:00:00Z",
            "originalMessageSender": {
                "user": {
                    "id": _SENDER_ID,
                    "displayName": "Alice",
                    "userIdentityType": "aadUser",
                }
            },
        }
    )
    msg = parse_msgraph_teams_channel_message(
        _active_message_payload(
            attachments=[
                {
                    "id": "att-fwd",
                    "contentType": "forwardedMessageReference",
                    "content": forwarded_content,
                }
            ]
        ),
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.attachments[0].forwarded_message is not None
    assert msg.attachments[0].forwarded_message.original_sender is not None
    assert msg.attachments[0].forwarded_message.original_sender.identity_type == "aadUser"


# --- Teams Channel reply messages ---


def test_parse_active_reply_message() -> None:
    msg = parse_msgraph_teams_channel_message(
        _active_reply_payload(),
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.message_kind is MsGraphTeamsChannelMessageKind.REPLY
    assert msg.remote_id == _REPLY_MESSAGE_ID
    assert msg.thread_root_remote_id == _ROOT_MESSAGE_ID
    assert msg.state is MsGraphTeamsChannelMessageState.ACTIVE
    assert msg.body_content == "Reply body"


def test_parse_deleted_reply_message() -> None:
    msg = parse_msgraph_teams_channel_message(
        _active_reply_payload(
            deletedDateTime="2024-01-01T12:00:00Z",
            lastModifiedDateTime="2024-01-01T12:00:00Z",
        ),
        expected_team_id=_TEAM_ID,
        expected_channel_id=_CHANNEL_ID,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
        max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert msg.state is MsGraphTeamsChannelMessageState.DELETED
    assert msg.deleted_at is not None
    assert msg.body_kind is None


def test_parse_reply_requires_reply_to_id() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_reply_payload(replyToId=None),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_reply_reply_to_id_must_match_expected_root() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_reply_payload(replyToId=_OTHER_ROOT_MESSAGE_ID),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_reply_id_must_differ_from_root_id() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_reply_payload(id=_ROOT_MESSAGE_ID, replyToId=_ROOT_MESSAGE_ID),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_reply_channel_identity_team_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_reply_payload(channelIdentity={"teamId": _OTHER_TEAM_ID, "channelId": _CHANNEL_ID}),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_reply_channel_identity_channel_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_reply_payload(channelIdentity={"teamId": _TEAM_ID, "channelId": _OTHER_CHANNEL_ID}),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_parse_reply_non_null_chat_id_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_channel_message(
            _active_reply_payload(chatId="19:chat@thread.v2"),
            expected_team_id=_TEAM_ID,
            expected_channel_id=_CHANNEL_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            expected_thread_root_remote_id=_ROOT_MESSAGE_ID,
            max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_validate_reply_page_returns_new_instances() -> None:
    original = _valid_reply_page()
    validated = validate_msgraph_teams_channel_reply_page(
        original,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=_ROOT_MESSAGE_ID,
        root_message_revision=_ETAG,
        graph_base_url=_GRAPH_BASE,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]
    assert validated.items[0].message_kind is MsGraphTeamsChannelMessageKind.REPLY


def test_reader_reply_request_sequence_and_params() -> None:
    http = MagicMock()
    _setup_replies_page_http(http)
    root = _valid_active_message()
    page = _reader(http).read_teams_channel_replies_page(
        root_message=root,
        continuation=None,
        limit=25,
        max_chars_per_message=1000,
    )
    assert http.get.call_count == 3
    pre_observation = http.get.call_args_list[0]
    collection = http.get.call_args_list[1]
    post_observation = http.get.call_args_list[2]
    assert pre_observation.args[0] == _ROOT_OBSERVATION_PATH
    assert collection.args[0] == _REPLIES_COLLECTION_PATH
    assert post_observation.args[0] == _ROOT_OBSERVATION_PATH
    assert collection.kwargs["params"] == {"$top": 25}
    assert "$filter" not in collection.kwargs["params"]
    assert "$orderby" not in collection.kwargs["params"]
    assert "$select" not in collection.kwargs["params"]
    assert "$expand" not in collection.kwargs["params"]
    assert collection.kwargs["headers"] == _PREFER
    assert page.items[0].message_kind is MsGraphTeamsChannelMessageKind.REPLY
    assert page.items[0].thread_root_remote_id == _ROOT_MESSAGE_ID


def test_reader_empty_reply_page_is_complete() -> None:
    http = MagicMock()
    _setup_replies_page_http(http, reply_items=[])
    page = _reader(http).read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert page.items == ()
    assert page.is_complete is True


def test_reader_one_reply_page() -> None:
    http = MagicMock()
    _setup_replies_page_http(http)
    page = _reader(http).read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert page.items[0].remote_id == _REPLY_MESSAGE_ID


def test_reader_multiple_replies_page() -> None:
    http = MagicMock()
    second_reply = _active_reply_payload(
        id="reply-msg-003",
        etag="reply-etag-3",
        body={"contentType": "text", "content": "Second reply"},
    )
    _setup_replies_page_http(http, reply_items=[_active_reply_payload(), second_reply])
    page = _reader(http).read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 2


def test_reader_reply_page_with_next_page_continuation() -> None:
    http = MagicMock()
    next_link = _replies_next_link()
    _setup_replies_page_http(http, next_link=next_link)
    page = _reader(http).read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert page.continuation is not None
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert page.is_complete is False


def test_reader_duplicate_reply_ids_last_occurrence_wins() -> None:
    http = MagicMock()
    first = _active_reply_payload(body={"contentType": "text", "content": "first"})
    second = _active_reply_payload(body={"contentType": "text", "content": "second"})
    _setup_replies_page_http(http, reply_items=[first, second])
    page = _reader(http).read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert page.items[0].body_content == "second"


def test_reader_rejects_root_message_in_reply_page() -> None:
    http = MagicMock()
    _setup_replies_page_http(http, reply_items=[_active_message_payload()])
    with pytest.raises(ValueError, match=_SAFE):
        _reader(http).read_teams_channel_replies_page(
            root_message=_valid_active_message(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_reader_rejects_reply_bound_to_other_root() -> None:
    http = MagicMock()
    wrong_reply = _active_reply_payload(replyToId=_OTHER_ROOT_MESSAGE_ID)
    _setup_replies_page_http(http, reply_items=[wrong_reply])
    with pytest.raises(ValueError, match=_SAFE):
        _reader(http).read_teams_channel_replies_page(
            root_message=_valid_active_message(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_validate_replies_continuation_accepts_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_replies_next_link(),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=_ROOT_MESSAGE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.url == continuation.url


def test_validate_replies_continuation_accepts_odata_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_replies_next_link(),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=_ROOT_MESSAGE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_replies_continuation_accepts_literal_with_escaped_quotes() -> None:
    root_id = "root'quote'part"
    channel_id = "channel'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_replies_next_link(root_id=root_id, channel_id=channel_id),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=channel_id,
        root_message_remote_id=root_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_replies_continuation_accepts_uppercase_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/TEAMS/{_QUOTED_TEAM_ID}/CHANNELS/"
            f"{_QUOTED_CHANNEL}/MESSAGES/{_QUOTED_ROOT_MESSAGE_ID}/REPLIES?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=_ROOT_MESSAGE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_replies_continuation_accepts_percent_encoded_root_literal() -> None:
    root_id = "root/special"
    encoded = quote(root_id, safe="")
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
            f"{_QUOTED_CHANNEL}/messages('{encoded}')/replies?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=root_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "root_id",
    [
        "opaque-messages-replies",
        "opaque-replies-only",
        "opaque-members-replies",
        "opaque-hostedContents-replies",
        "opaque-teams-channels",
        "opaque-channels-messages",
        "opaque-users-chats",
        "opaque-chats-replies",
        "messages",
        "replies",
        "members",
        "hostedContents",
        "teams",
        "channels",
        "users",
        "chats",
    ],
)
def test_validate_replies_continuation_accepts_opaque_root_id_with_reserved_words(
    root_id: str,
) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_slash_replies_next_link(root_id),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=root_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_replies_continuation_accepts_odata_literal_with_quote_percent() -> None:
    root_id = "root'delta/special"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_percent_encoded_replies_next_link(root_id),
    )
    validated = validate_msgraph_teams_channel_replies_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        root_message_remote_id=root_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{quote(_OTHER_CHANNEL_ID, safe='')}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quote(_OTHER_ROOT_MESSAGE_ID, safe='')}/replies?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/{_QUOTED_REPLY_MESSAGE_ID}?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/extra?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/delta?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/children?$skiptoken=x",
    ],
)
def test_validate_replies_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT) as exc:
        validate_msgraph_teams_channel_replies_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            root_message_remote_id=_ROOT_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _TEAM_ID not in str(exc.value)


def test_validate_replies_continuation_rejects_delta_kind() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_replies_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT):
        validate_msgraph_teams_channel_replies_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            root_message_remote_id=_ROOT_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_replies_next_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="next_page",
            url=_replies_next_link(),
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=123,
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url="",
        ),
    ],
)
def test_validate_replies_continuation_rejects_model_construct_malformed(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT) as exc:
        validate_msgraph_teams_channel_replies_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            root_message_remote_id=_ROOT_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _ROOT_MESSAGE_ID not in str(exc.value)


class _CountingRepliesClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChannelReplyPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page
        self.call_count = 0
        self.last_continuation: MsGraphKnowledgeContinuation | None = None

    def read_teams_channel_replies_page(
        self,
        *,
        root_message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelReplyPage:
        self.call_count += 1
        self.last_continuation = continuation
        return self._custom_page


class _CustomRepliesSuite(CollaborationSuite):
    def __init__(self, client: _CountingRepliesClient) -> None:
        self._client = client

    def read_teams_channel_replies_page(
        self,
        *,
        root_message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChannelReplyPage:
        return self._client.read_teams_channel_replies_page(
            root_message=root_message,
            continuation=continuation,
            limit=limit,
            max_chars_per_message=max_chars_per_message,
        )

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


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_replies_next_link(),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_replies_next_link(team_id=_OTHER_TEAM_ID),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_replies_next_link(root_id=_OTHER_ROOT_MESSAGE_ID),
        ),
    ],
)
def test_integration_rejects_malformed_reply_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    client = _CountingRepliesClient(page=_valid_reply_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT) as exc:
        integration.read_teams_channel_replies_page(
            root_message=_valid_active_message(),
            continuation=continuation,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


def test_integration_valid_reply_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_replies_next_link(),
    )
    client = _CountingRepliesClient(page=_valid_reply_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=continuation,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not _valid_reply_page().items[0]
    assert returned.items[0].message_kind is MsGraphTeamsChannelMessageKind.REPLY


def test_read_and_validate_reply_observation_uses_reply_endpoint() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_reply_payload())
    transport = MsGraphKnowledgeTransport(config=_config(), http_client=http)
    read_and_validate_current_teams_channel_message_observation(
        message=_valid_active_reply(),
        transport=transport,
    )
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _REPLY_OBSERVATION_PATH
    assert http.get.call_args.kwargs["headers"] == _PREFER


@pytest.mark.parametrize(
    ("payload_overrides", "message_overrides"),
    [
        ({"id": "wrong-reply-id"}, {}),
        ({"replyToId": _OTHER_ROOT_MESSAGE_ID}, {}),
        ({"etag": "wrong-etag"}, {}),
        ({"deletedDateTime": "2024-01-01T12:00:00Z"}, {}),
        (
            {"channelIdentity": {"teamId": _OTHER_TEAM_ID, "channelId": _CHANNEL_ID}},
            {},
        ),
        (
            {"channelIdentity": {"teamId": _TEAM_ID, "channelId": _OTHER_CHANNEL_ID}},
            {},
        ),
    ],
)
def test_read_and_validate_reply_observation_failures(
    payload_overrides: dict[str, object],
    message_overrides: dict[str, object],
) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_reply_payload(**payload_overrides),
    )
    transport = MsGraphKnowledgeTransport(config=_config(), http_client=http)
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        read_and_validate_current_teams_channel_message_observation(
            message=_valid_active_reply(**message_overrides),
            transport=transport,
        )
    assert http.get.call_args.args[0] == _REPLY_OBSERVATION_PATH
    assert f"/messages/{_QUOTED_REPLY_MESSAGE_ID}" not in http.get.call_args.args[0].replace(
        f"/replies/{_QUOTED_REPLY_MESSAGE_ID}", ""
    )


def test_read_and_validate_reply_observation_does_not_use_flat_message_path() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_reply_payload())
    transport = MsGraphKnowledgeTransport(config=_config(), http_client=http)
    read_and_validate_current_teams_channel_message_observation(
        message=_valid_active_reply(),
        transport=transport,
    )
    called_path = http.get.call_args.args[0]
    assert called_path == _REPLY_OBSERVATION_PATH
    assert called_path != f"/teams/{_QUOTED_TEAM_ID}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_REPLY_MESSAGE_ID}"


# --- root message reference ---


def _valid_root_reference(**overrides: object) -> MsGraphTeamsChannelRootMessageReference:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelRootMessageReference(**defaults)


def test_root_reference_active_valid() -> None:
    ref = _valid_root_reference()
    assert ref.state is MsGraphTeamsChannelMessageState.ACTIVE


def test_root_reference_deleted_valid() -> None:
    ref = _valid_root_reference(state=MsGraphTeamsChannelMessageState.DELETED)
    assert ref.state is MsGraphTeamsChannelMessageState.DELETED


def test_root_reference_state_enum_accepted() -> None:
    ref = validate_msgraph_teams_channel_root_message_reference(
        {
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "remote_id": _MESSAGE_ID,
            "revision": _ETAG,
            "state": MsGraphTeamsChannelMessageState.ACTIVE,
        }
    )
    assert ref.state is MsGraphTeamsChannelMessageState.ACTIVE


@pytest.mark.parametrize("raw_state", ["active", "deleted"])
def test_root_reference_raw_state_strings_rejected(raw_state: str) -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": _TEAM_ID,
                "channel_remote_id": _CHANNEL_ID,
                "remote_id": _MESSAGE_ID,
                "revision": _ETAG,
                "state": raw_state,
            }
        )
    assert exc.value.__cause__ is None
    assert raw_state not in str(exc.value)


def test_root_reference_invalid_team_id() -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": "",
                "channel_remote_id": _CHANNEL_ID,
                "remote_id": _MESSAGE_ID,
                "revision": _ETAG,
                "state": MsGraphTeamsChannelMessageState.ACTIVE,
            }
        )
    assert exc.value.__cause__ is None
    assert _TEAM_ID not in str(exc.value)


def test_root_reference_invalid_channel_id() -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": _TEAM_ID,
                "channel_remote_id": "bad\x00id",
                "remote_id": _MESSAGE_ID,
                "revision": _ETAG,
                "state": MsGraphTeamsChannelMessageState.ACTIVE,
            }
        )
    assert exc.value.__cause__ is None


def test_root_reference_invalid_root_id() -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": _TEAM_ID,
                "channel_remote_id": _CHANNEL_ID,
                "remote_id": "",
                "revision": _ETAG,
                "state": MsGraphTeamsChannelMessageState.ACTIVE,
            }
        )
    assert exc.value.__cause__ is None
    assert _MESSAGE_ID not in str(exc.value)


def test_root_reference_invalid_revision() -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": _TEAM_ID,
                "channel_remote_id": _CHANNEL_ID,
                "remote_id": _MESSAGE_ID,
                "revision": "",
                "state": MsGraphTeamsChannelMessageState.ACTIVE,
            }
        )
    assert exc.value.__cause__ is None
    assert _ETAG not in str(exc.value)


def test_root_reference_oversized_revision() -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": _TEAM_ID,
                "channel_remote_id": _CHANNEL_ID,
                "remote_id": _MESSAGE_ID,
                "revision": "x" * 5000,
                "state": MsGraphTeamsChannelMessageState.ACTIVE,
            }
        )
    assert exc.value.__cause__ is None


def test_root_reference_control_characters_in_revision() -> None:
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(
            {
                "team_remote_id": _TEAM_ID,
                "channel_remote_id": _CHANNEL_ID,
                "remote_id": _MESSAGE_ID,
                "revision": "bad\x00etag",
                "state": MsGraphTeamsChannelMessageState.ACTIVE,
            }
        )
    assert exc.value.__cause__ is None


def test_root_reference_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MsGraphTeamsChannelRootMessageReference(
            team_remote_id=_TEAM_ID,
            channel_remote_id=_CHANNEL_ID,
            remote_id=_MESSAGE_ID,
            revision=_ETAG,
            state=MsGraphTeamsChannelMessageState.ACTIVE,
            body_content="secret",
        )


def test_root_reference_model_construct_deep_validation() -> None:
    malformed = MsGraphTeamsChannelRootMessageReference.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state="active",
    )
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_channel_root_message_reference(malformed)
    assert exc.value.__cause__ is None


def test_root_reference_repr_hides_identifiers() -> None:
    ref = _valid_root_reference()
    rendered = repr(ref)
    assert _TEAM_ID not in rendered
    assert _CHANNEL_ID not in rendered
    assert _MESSAGE_ID not in rendered
    assert _ETAG not in rendered
    assert "active" in rendered.lower() or "ACTIVE" in rendered


def test_root_reference_attributes_accessible() -> None:
    ref = _valid_root_reference()
    assert ref.team_remote_id == _TEAM_ID
    assert ref.channel_remote_id == _CHANNEL_ID
    assert ref.remote_id == _MESSAGE_ID
    assert ref.revision == _ETAG


def test_root_reference_serialization_complete() -> None:
    ref = _valid_root_reference()
    dumped = ref.model_dump()
    assert dumped["team_remote_id"] == _TEAM_ID
    assert dumped["channel_remote_id"] == _CHANNEL_ID
    assert dumped["remote_id"] == _MESSAGE_ID
    assert dumped["revision"] == _ETAG
    assert dumped["state"] is MsGraphTeamsChannelMessageState.ACTIVE


def test_root_message_reference_from_message() -> None:
    root = _valid_active_message()
    ref = root_message_reference_from_message(root)
    assert ref.remote_id == _MESSAGE_ID
    assert ref.revision == _ETAG
    assert ref.state is MsGraphTeamsChannelMessageState.ACTIVE
