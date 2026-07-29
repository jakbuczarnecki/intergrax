# © Artur Czarnecki. All rights reserved.

"""Unit tests for Microsoft Graph Teams Channel knowledge-read messages surface."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

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
    ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    MsGraphTeamsChatAttachmentKind,
    MsGraphTeamsChatBodyKind,
    MsGraphTeamsChatImportance,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageType,
    MsGraphTeamsIdentityKind,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    parse_msgraph_teams_chat,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageState,
    MsGraphTeamsChatMessageWindow,
    MsGraphTeamsChatMessagesReader,
    MsGraphTeamsForwardedMessageReference,
    MsGraphTeamsIdentity,
    format_msgraph_teams_chat_window_datetime,
    parse_msgraph_teams_chat_message,
    validate_msgraph_teams_chat_attachment_reference,
    validate_msgraph_teams_chat_message,
    validate_msgraph_teams_chat_message_snapshot_page,
    validate_msgraph_teams_chat_messages_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX = "user@contoso.com"
_OTHER_MAILBOX = "other@contoso.com"
_CHAT_ID = "19:chat-abc@thread.v2"
_OTHER_CHAT_ID = "19:other-chat@thread.v2"
_MESSAGE_ID = "msg-001"
_SENDER_ID = "sender-secret-id"
_ETAG = "etag-1"
_SECRET_TOKEN = "secret-skiptoken-value"
_QUOTED_MAILBOX = quote(_MAILBOX, safe="")
_QUOTED_OTHER_MAILBOX = quote(_OTHER_MAILBOX, safe="")
_QUOTED_CHAT = quote(_CHAT_ID, safe="")
_PREFER = {"Prefer": "include-unknown-enum-members"}
_SAFE = "unexpected Microsoft Graph Teams chat messages response"
_CONT = "invalid Microsoft Graph Teams chat messages continuation"
_REQUEST_ERROR = "invalid Microsoft Graph Teams chat messages request"
_VALIDATION_ERROR = "Microsoft Graph Teams Chat validation is not configured"


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
    msg = parse_msgraph_teams_chat_message(
        payload,
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
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
    msg = parse_msgraph_teams_chat_message(
        payload,
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
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
    msg = parse_msgraph_teams_chat_message(
        payload,
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert msg.sender is not None
    assert msg.sender.identity_type == "futureProviderValue"


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS + 1, True, False, "100", 1.5, None],
)
def test_validate_message_rejects_invalid_max_chars(max_chars: object) -> None:
    message = _valid_active_message()
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message, max_chars=max_chars)  # type: ignore[arg-type]


def test_validate_message_body_at_max_chars_succeeds() -> None:
    body = "x" * 100
    message = _valid_active_message(body_content=body)
    validated = validate_msgraph_teams_chat_message(message, max_chars=100)
    assert validated.body_content == body
    assert validated is not message


def test_validate_message_body_one_over_max_chars_fails() -> None:
    message = _valid_active_message(body_content="x" * 101)
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message, max_chars=100)


def test_validate_message_body_with_nul_fails() -> None:
    message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
        body_content="a\x00b",
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message, max_chars=100)


def test_validate_message_missing_body_content_rejected() -> None:
    message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message)


def test_validate_message_integer_body_content_rejected() -> None:
    message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
        body_content=123,
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message)


def test_validate_message_malformed_sender_rejected() -> None:
    message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
        body_content="hello",
        sender=MsGraphTeamsIdentity.model_construct(),
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message)


def test_validate_message_malformed_attachment_rejected() -> None:
    message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
        body_content="hello",
        attachments=(
            MsGraphTeamsChatAttachmentReference.model_construct(
                attachment_kind="reference",
                content_type="reference",
            ),
        ),
    )
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_message(message)


def test_parse_body_exactly_at_max_chars() -> None:
    body = "a" * 50
    msg = parse_msgraph_teams_chat_message(
        _active_message_payload(body={"contentType": "text", "content": body}),
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=50,
    )
    assert msg.body_content == body


def test_parse_body_one_over_max_chars_fails() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_chat_message(
            _active_message_payload(body={"contentType": "text", "content": "a" * 51}),
            expected_mailbox_user_id=_MAILBOX,
            expected_chat_id=_CHAT_ID,
            max_chars=50,
        )


def test_parse_body_with_nul_fails() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        parse_msgraph_teams_chat_message(
            _active_message_payload(body={"contentType": "text", "content": "a\x00b"}),
            expected_mailbox_user_id=_MAILBOX,
            expected_chat_id=_CHAT_ID,
            max_chars=100,
        )


@pytest.mark.parametrize(
    ("last_modified", "created_at"),
    [
        (datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), datetime(2023, 12, 31, 12, 0, tzinfo=timezone.utc)),
        (datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc), datetime(2023, 12, 31, 10, 0, tzinfo=timezone.utc)),
        (datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc), datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)),
        (datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc), datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)),
    ],
)
def test_snapshot_page_rejects_message_outside_window(
    last_modified: datetime,
    created_at: datetime,
) -> None:
    message = _valid_active_message(
        created_at=created_at,
        last_modified_at=last_modified,
    )
    page = MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        window=_window(),
        items=(message,),
    )
    with pytest.raises(ValueError, match=_SAFE) as exc:
        validate_msgraph_teams_chat_message_snapshot_page(
            page,
            chat=_chat(),
            window=_window(),
            graph_base_url=_GRAPH_BASE,
            max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )
    assert str(last_modified) not in str(exc.value)
    assert _CHAT_ID not in str(exc.value)
    assert "Hello" not in str(exc.value)


def test_snapshot_page_accepts_message_inside_window() -> None:
    message = _valid_active_message()
    page = MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        window=_window(),
        items=(message,),
    )
    validated = validate_msgraph_teams_chat_message_snapshot_page(
        page,
        chat=_chat(),
        window=_window(),
        graph_base_url=_GRAPH_BASE,
        max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert len(validated.items) == 1


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
        match="unexpected Microsoft Graph Teams chat messages response",
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
                    original_chat_id=_CHAT_ID,
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
    with pytest.raises(ValueError, match=_SAFE) as exc:
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
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(content_url=None)
        )


def test_attachment_reference_http_url_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(content_url="http://insecure.example/file")
        )


def test_attachment_reference_userinfo_url_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(content_url="https://user:pass@contoso.example/file")
        )


def test_attachment_reference_with_embedded_content_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(embedded_content="inline")
        )


def test_attachment_forwarded_valid() -> None:
    forwarded = MsGraphTeamsForwardedMessageReference(
        original_message_id="orig-msg",
        original_chat_id=_CHAT_ID,
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
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            MsGraphTeamsChatAttachmentReference(
                remote_id="att-fwd",
                attachment_kind=MsGraphTeamsChatAttachmentKind.FORWARDED_MESSAGE_REFERENCE,
                content_type="forwardedMessageReference",
                has_thumbnail_url=False,
            )
        )


def test_attachment_kind_inconsistent_with_content_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
        validate_msgraph_teams_chat_attachment_reference(
            _reference_attachment(
                attachment_kind=MsGraphTeamsChatAttachmentKind.CARD,
            )
        )


def test_attachment_card_with_content_url_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE):
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


def test_invalid_continuation_rejected_before_messages_request() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/messages?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT):
        _reader(http).read_messages_snapshot_page(
            chat=_chat(),
            window=_window(),
            continuation=continuation,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )
    http.get.assert_not_called()


class _CountingMessagesClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChatMessageSnapshotPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page
        self.call_count = 0
        self.last_continuation: MsGraphKnowledgeContinuation | None = None

    def read_teams_chat_messages_snapshot_page(
        self,
        *,
        chat: object,
        window: object,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
        self.call_count += 1
        self.last_continuation = continuation
        return self._custom_page


class _CustomMessagesSuite(CollaborationSuite):
    def __init__(self, client: _CountingMessagesClient) -> None:
        self._client = client

    def read_teams_chat_messages_snapshot_page(
        self,
        *,
        chat: object,
        window: object,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
        max_chars_per_message: int = DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
        return self._client.read_teams_chat_messages_snapshot_page(
            chat=chat,
            window=window,
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


def _valid_snapshot_page() -> MsGraphTeamsChatMessageSnapshotPage:
    return MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        window=_window(),
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
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
                f"{_QUOTED_CHAT}/messages/delta?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=(
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats/"
                f"{_QUOTED_CHAT}/messages?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=(
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
                f"{quote(_OTHER_CHAT_ID, safe='')}/messages?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
    ],
)
def test_integration_rejects_malformed_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    client = _CountingMessagesClient(page=_valid_snapshot_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT) as exc:
        integration.read_teams_chat_messages_snapshot_page(
            chat=_chat(),
            window=_window(),
            continuation=continuation,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS + 1, True, False, "100", 1.5, None],
)
def test_integration_rejects_invalid_max_chars_before_custom_call(max_chars: object) -> None:
    client = _CountingMessagesClient(page=_valid_snapshot_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_teams_chat_messages_snapshot_page(
            chat=_chat(),
            window=_window(),
            continuation=None,
            limit=50,
            max_chars_per_message=max_chars,  # type: ignore[arg-type]
        )
    assert client.call_count == 0


def test_integration_valid_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/messages?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    client = _CountingMessagesClient(page=_valid_snapshot_page(), http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_chat_messages_snapshot_page(
        chat=_chat(),
        window=_window(),
        continuation=continuation,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not _valid_snapshot_page().items[0]


def test_integration_custom_client_raw_attachment_kind_rejected() -> None:
    message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        remote_id=_MESSAGE_ID,
        revision=_ETAG,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        last_modified_at=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
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
    page = MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        window=_window(),
        items=(message,),
    )
    client = _CountingMessagesClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE) as exc:
        integration.read_teams_chat_messages_snapshot_page(
            chat=_chat(),
            window=_window(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )
    assert client.call_count == 1
    assert exc.value.__cause__ is None
    error_text = str(exc.value)
    assert "Hello" not in error_text
    assert "https://contoso.example/file" not in error_text
    assert "att-raw-kind" not in error_text
    assert _CHAT_ID not in error_text


def test_integration_custom_client_body_above_requested_limit_rejected() -> None:
    oversized = _valid_active_message(body_content="x" * 200)
    page = MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX,
        chat_remote_id=_CHAT_ID,
        window=_window(),
        items=(oversized,),
    )
    client = _CountingMessagesClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE):
        integration.read_teams_chat_messages_snapshot_page(
            chat=_chat(),
            window=_window(),
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


def test_parse_forwarded_message_reference_from_provider_payload() -> None:
    forwarded_content = json.dumps(
        {
            "originalMessageId": "orig-msg",
            "originalConversationId": _CHAT_ID,
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
    msg = parse_msgraph_teams_chat_message(
        _active_message_payload(
            attachments=[
                {
                    "id": "att-fwd",
                    "contentType": "forwardedMessageReference",
                    "content": forwarded_content,
                }
            ]
        ),
        expected_mailbox_user_id=_MAILBOX,
        expected_chat_id=_CHAT_ID,
        max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    )
    assert msg.attachments[0].forwarded_message is not None
    assert msg.attachments[0].forwarded_message.original_sender is not None
    assert msg.attachments[0].forwarded_message.original_sender.identity_type == "aadUser"
