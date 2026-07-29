# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Mail knowledge-read content surface."""

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
    ABSOLUTE_MAIL_CONTENT_MAX_CHARS,
    DEFAULT_MAIL_CONTENT_MAX_CHARS,
    MsGraphKnowledgeTransport,
    MsGraphMailContentReader,
    MsGraphMailContentTooLarge,
    MsGraphMailImportance,
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    MsGraphMailMessageChanged,
    MsGraphMailMessageContent,
    parse_msgraph_mail_participant,
    validate_msgraph_mail_message_content,
    validate_msgraph_mail_participant,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_FOLDER_ID = "folder-abc-123"
_OTHER_FOLDER_ID = "other-folder"
_MESSAGE_ID = "AAMkAGI2THSAAA-immutable-opaque-id"
_OTHER_MESSAGE_ID = "AAMkAGI2THSBBB"
_CHANGE_KEY = "change-key-secret-value"
_OTHER_CHANGE_KEY = "other-change-key"
_CONVERSATION_ID = "conv-abc"
_INTERNET_MESSAGE_ID = "<msg-id@example.com>"
_SUBJECT = "Quarterly report"
_SECRET_SUBJECT = "secret-subject-value"
_SECRET_BODY = "secret-body-text-value"
_SECRET_ADDRESS = "secret-participant@example.com"
_SECRET_DISPLAY_NAME = "Secret Participant Name"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_MESSAGE_ID = quote(_MESSAGE_ID, safe="")
_CONTENT_PATH = f"/users/{_QUOTED_MAILBOX}/messages/{_QUOTED_MESSAGE_ID}"
_SELECT = (
    "id,parentFolderId,changeKey,conversationId,internetMessageId,subject,body,uniqueBody,"
    "from,sender,replyTo,toRecipients,ccRecipients,bccRecipients,createdDateTime,"
    "lastModifiedDateTime,receivedDateTime,sentDateTime,isRead,isDraft,hasAttachments,importance"
)
_TEXT_BODY_HEADERS = {
    "Prefer": 'IdType="ImmutableId", outlook.body-content-type="text"',
}
_SAFE_ERROR = "unexpected Microsoft Graph Mail content response"
_KNOWLEDGE_SAFE_ERROR = "unexpected Microsoft Graph knowledge response"
_MESSAGES_SAFE_ERROR = "unexpected Microsoft Graph mail messages response"
_REQUEST_ERROR = "invalid Microsoft Graph Mail content request"
_CHANGED_ERROR = "Microsoft Graph Mail message changed during read"
_TOO_LARGE_ERROR = "Microsoft Graph Mail message exceeds the configured content limit"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Mail content capability"
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


def _text_body(content: str) -> dict[str, str]:
    return {"contentType": "text", "content": content}


def _participant(
    address: str,
    *,
    display_name: str | None = None,
    include_name_key: bool = True,
) -> dict[str, Any]:
    email_address: dict[str, Any] = {"address": address}
    if include_name_key:
        if display_name is not None:
            email_address["name"] = display_name
    return {"emailAddress": email_address}


def _content_payload(
    *,
    message_id: str = _MESSAGE_ID,
    parent_folder_id: str = _FOLDER_ID,
    change_key: str = _CHANGE_KEY,
    conversation_id: str | None = None,
    internet_message_id: str | None = None,
    subject: str | None = None,
    include_subject_key: bool = False,
    subject_null: bool = False,
    body_text: str = "Hello, world.",
    include_body: bool = True,
    unique_body_text: str | None = None,
    include_unique_body: bool = False,
    unique_body_null: bool = False,
    from_participant: dict[str, Any] | None = None,
    sender_participant: dict[str, Any] | None = None,
    include_from: bool = True,
    include_sender: bool = True,
    reply_to: list[dict[str, Any]] | None = None,
    to_recipients: list[dict[str, Any]] | None = None,
    cc_recipients: list[dict[str, Any]] | None = None,
    bcc_recipients: list[dict[str, Any]] | None = None,
    removed: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": message_id,
        "parentFolderId": parent_folder_id,
        "changeKey": change_key,
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
        "isRead": False,
        "isDraft": False,
        "hasAttachments": False,
        "importance": "normal",
        "replyTo": [] if reply_to is None else reply_to,
        "toRecipients": [] if to_recipients is None else to_recipients,
        "ccRecipients": [] if cc_recipients is None else cc_recipients,
        "bccRecipients": [] if bcc_recipients is None else bcc_recipients,
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    if internet_message_id is not None:
        payload["internetMessageId"] = internet_message_id
    if include_subject_key:
        payload["subject"] = None if subject_null else (subject if subject is not None else "")
    elif subject is not None:
        payload["subject"] = subject
    if include_body:
        payload["body"] = _text_body(body_text)
    if include_unique_body:
        if unique_body_null:
            payload["uniqueBody"] = None
        elif unique_body_text is not None:
            payload["uniqueBody"] = _text_body(unique_body_text)
    if include_from:
        payload["from"] = from_participant
    if include_sender:
        payload["sender"] = sender_participant
    if removed:
        payload["@removed"] = {"reason": "deleted"}
    return payload


def _valid_active_change_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "scope_folder_id": _FOLDER_ID,
        "remote_id": _MESSAGE_ID,
        "kind": MsGraphMailMessageChangeKind.ACTIVE,
        "parent_folder_id": _FOLDER_ID,
        "change_key": _CHANGE_KEY,
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        "is_read": False,
        "is_draft": False,
        "has_attachments": False,
        "importance": MsGraphMailImportance.NORMAL,
    }
    defaults.update(overrides)
    return defaults


def _valid_active_change(**overrides: object) -> MsGraphMailMessageChange:
    return MsGraphMailMessageChange(**_valid_active_change_kwargs(**overrides))


def _valid_removed_change(**overrides: object) -> MsGraphMailMessageChange:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "scope_folder_id": _FOLDER_ID,
        "remote_id": _MESSAGE_ID,
        "kind": MsGraphMailMessageChangeKind.REMOVED,
        "removed_reason": "deleted",
    }
    defaults.update(overrides)
    return MsGraphMailMessageChange(**defaults)


def _valid_message_content(**overrides: object) -> MsGraphMailMessageContent:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _MESSAGE_ID,
        "parent_folder_id": _FOLDER_ID,
        "content_revision": _CHANGE_KEY,
        "body_text": "Hello, world.",
    }
    defaults.update(overrides)
    return MsGraphMailMessageContent(**defaults)


def _reader(http: MagicMock) -> MsGraphMailContentReader:
    return MsGraphMailContentReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _setup_happy_path(
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[MagicMock, MsGraphMailContentReader]:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload or _content_payload())
    return http, _reader(http)


# --- constants ---


def test_default_max_chars_constants() -> None:
    assert DEFAULT_MAIL_CONTENT_MAX_CHARS == 2_000_000
    assert ABSOLUTE_MAIL_CONTENT_MAX_CHARS == 8_000_000


# --- participant parser ---


def test_parse_participant_with_display_name() -> None:
    participant = parse_msgraph_mail_participant(
        _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    assert participant.address == _SECRET_ADDRESS
    assert participant.display_name == _SECRET_DISPLAY_NAME


def test_parse_participant_without_name_key() -> None:
    participant = parse_msgraph_mail_participant(
        _participant(_SECRET_ADDRESS, include_name_key=False)
    )
    assert participant.display_name is None


def test_parse_participant_empty_name_becomes_none() -> None:
    participant = parse_msgraph_mail_participant(
        _participant(_SECRET_ADDRESS, display_name="   ")
    )
    assert participant.display_name is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"emailAddress": None},
        {"emailAddress": {}},
        {"emailAddress": {"name": "Only Name"}},
        {"emailAddress": {"address": ""}},
        {"emailAddress": {"address": "a@b.com", "name": 123}},
        {"emailAddress": {"address": "a\x00@b.com"}},
    ],
)
def test_parse_participant_malformed(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_mail_participant(payload)
    assert exc.value.__cause__ is None
    assert _SECRET_ADDRESS not in str(exc.value)


def test_validate_participant_returns_new_instance() -> None:
    original = parse_msgraph_mail_participant(
        _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    validated = validate_msgraph_mail_participant(original)
    assert validated == original
    assert validated is not original


# --- success: body and uniqueBody ---


def test_read_text_body() -> None:
    http, reader = _setup_happy_path(
        payload=_content_payload(body_text=_SECRET_BODY),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.body_text == _SECRET_BODY
    assert result.unique_body_text is None
    http.get.assert_called_once()


def test_read_empty_body() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(body_text=""))
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.body_text == ""


def test_read_unique_body_text() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(
            include_unique_body=True,
            unique_body_text="Unique portion",
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.unique_body_text == "Unique portion"


def test_read_unique_body_absent() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(include_unique_body=False))
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.unique_body_text is None


def test_read_unique_body_null() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(include_unique_body=True, unique_body_null=True),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.unique_body_text is None


# --- success: from/sender variants ---


def test_read_from_participant_only() -> None:
    from_payload = _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    _, reader = _setup_happy_path(
        payload=_content_payload(
            from_participant=from_payload,
            sender_participant=None,
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.from_participant is not None
    assert result.from_participant.address == _SECRET_ADDRESS
    assert result.sender_participant is None


def test_read_sender_participant_only() -> None:
    sender_payload = _participant("sender@contoso.com", display_name="Sender")
    _, reader = _setup_happy_path(
        payload=_content_payload(
            from_participant=None,
            sender_participant=sender_payload,
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.from_participant is None
    assert result.sender_participant is not None
    assert result.sender_participant.address == "sender@contoso.com"


def test_read_from_and_sender_same() -> None:
    participant = _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    _, reader = _setup_happy_path(
        payload=_content_payload(
            from_participant=participant,
            sender_participant=participant,
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.from_participant is not None
    assert result.sender_participant is not None
    assert result.from_participant.address == result.sender_participant.address


def test_read_from_and_sender_different() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(
            from_participant=_participant("from@contoso.com"),
            sender_participant=_participant("sender@contoso.com"),
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.from_participant is not None
    assert result.sender_participant is not None
    assert result.from_participant.address != result.sender_participant.address


# --- success: recipients ---


def test_read_all_recipient_lists() -> None:
    reply = _participant("reply@contoso.com")
    to_recipient = _participant("to@contoso.com", display_name="To User")
    cc_recipient = _participant("cc@contoso.com")
    bcc_recipient = _participant("bcc@contoso.com")
    _, reader = _setup_happy_path(
        payload=_content_payload(
            reply_to=[reply],
            to_recipients=[to_recipient],
            cc_recipients=[cc_recipient],
            bcc_recipients=[bcc_recipient],
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert len(result.reply_to) == 1
    assert len(result.to_recipients) == 1
    assert len(result.cc_recipients) == 1
    assert len(result.bcc_recipients) == 1
    assert result.to_recipients[0].display_name == "To User"


def test_read_empty_recipient_lists() -> None:
    _, reader = _setup_happy_path(payload=_content_payload())
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.reply_to == ()
    assert result.to_recipients == ()
    assert result.cc_recipients == ()
    assert result.bcc_recipients == ()


# --- success: subject and metadata ---


def test_read_subject_present() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(
            subject=_SECRET_SUBJECT,
            conversation_id=_CONVERSATION_ID,
            internet_message_id=_INTERNET_MESSAGE_ID,
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.subject == _SECRET_SUBJECT
    assert result.conversation_id == _CONVERSATION_ID
    assert result.internet_message_id == _INTERNET_MESSAGE_ID


def test_read_empty_subject() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(subject=""))
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.subject == ""


def test_read_null_subject() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(include_subject_key=True, subject_null=True),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.subject is None


def test_read_subject_absent() -> None:
    _, reader = _setup_happy_path(payload=_content_payload())
    result = reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert result.subject is None


# --- success: body at limits ---


def test_read_body_16001_chars_within_max_chars_succeeds() -> None:
    body = "x" * 16_001
    _, reader = _setup_happy_path(payload=_content_payload(body_text=body))
    result = reader.read_message_content(message=_valid_active_change(), max_chars=20_000)
    assert len(result.body_text) == 16_001


def test_read_body_at_max_chars_succeeds() -> None:
    body = "x" * 100_000
    _, reader = _setup_happy_path(payload=_content_payload(body_text=body))
    result = reader.read_message_content(message=_valid_active_change(), max_chars=100_000)
    assert len(result.body_text) == 100_000


def test_read_body_one_over_max_chars_rejected() -> None:
    body = "x" * 100_001
    http = MagicMock()
    http.get.return_value = _json_response(payload=_content_payload(body_text=body))
    with pytest.raises(MsGraphMailContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_active_change(), max_chars=100_000)
    assert exc.value.__cause__ is None


def test_read_combined_body_and_unique_at_max_chars_succeeds() -> None:
    body = "a" * 70_000
    unique = "b" * 30_000
    _, reader = _setup_happy_path(
        payload=_content_payload(
            body_text=body,
            include_unique_body=True,
            unique_body_text=unique,
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=100_000)
    assert len(result.body_text) + len(result.unique_body_text or "") == 100_000


def test_read_combined_body_and_unique_one_over_max_chars_rejected() -> None:
    body = "a" * 70_000
    unique = "b" * 30_001
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_content_payload(
            body_text=body,
            include_unique_body=True,
            unique_body_text=unique,
        ),
    )
    with pytest.raises(MsGraphMailContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_active_change(), max_chars=100_000)
    assert exc.value.__cause__ is None


def test_read_combined_body_at_limit() -> None:
    body = "a" * 1000
    unique = "b" * 1000
    _, reader = _setup_happy_path(
        payload=_content_payload(
            body_text=body,
            include_unique_body=True,
            unique_body_text=unique,
        ),
    )
    result = reader.read_message_content(message=_valid_active_change(), max_chars=2000)
    assert len(result.body_text) + len(result.unique_body_text or "") == 2000


# --- malformed provider responses ---


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        _content_payload(include_body=False),
        _content_payload() | {"body": None},
        _content_payload() | {"body": {"contentType": "html", "content": "x"}},
        _content_payload() | {"body": {"contentType": "text"}},
        _content_payload() | {"body": {"contentType": "text", "content": 1}},
        _content_payload() | {"body": {"contentType": "", "content": "x"}},
        _content_payload(body_text="bad\x00body"),
        _content_payload(include_unique_body=True, unique_body_text="x")
        | {"uniqueBody": "not-a-dict"},
        _content_payload(include_from=False),
        _content_payload(include_sender=False),
        _content_payload() | {"replyTo": None},
        _content_payload() | {"toRecipients": "not-a-list"},
        _content_payload() | {"ccRecipients": [None]},
        _content_payload(conversation_id=123),
        _content_payload(internet_message_id=123),
        _content_payload(subject=123),
        _content_payload() | {"id": _OTHER_MESSAGE_ID},
        _content_payload() | {"parentFolderId": _OTHER_FOLDER_ID},
        _content_payload() | {"changeKey": _OTHER_CHANGE_KEY},
        _content_payload() | {"lastModifiedDateTime": "2024-06-01T12:00:00"},
        _content_payload() | {"isRead": 1},
        _content_payload(removed=True),
    ],
)
def test_read_malformed_provider_payload(payload: object) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload)
    with pytest.raises(
        (ValueError, MsGraphMailMessageChanged),
        match=f"{_SAFE_ERROR}|{_KNOWLEDGE_SAFE_ERROR}|{_MESSAGES_SAFE_ERROR}|{_CHANGED_ERROR}",
    ) as exc:
        _reader(http).read_message_content(message=_valid_active_change(), max_chars=10_000)
    if isinstance(exc.value, ValueError):
        assert exc.value.__cause__ is None
    assert _SECRET_BODY not in str(exc.value)
    assert _SECRET_SUBJECT not in str(exc.value)


# --- consistency: identity mismatch and REMOVED ---


@pytest.mark.parametrize(
    "field_override",
    [
        {"message_id": _OTHER_MESSAGE_ID},
        {"parent_folder_id": _OTHER_FOLDER_ID},
        {"change_key": _OTHER_CHANGE_KEY},
    ],
)
def test_read_identity_mismatch_raises_changed(field_override: dict[str, str]) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_content_payload(**field_override))
    with pytest.raises(MsGraphMailMessageChanged, match=_CHANGED_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert exc.value.__cause__ is None
    assert _CHANGE_KEY not in str(exc.value)
    assert _MESSAGE_ID not in str(exc.value)


def test_removed_input_rejected_before_http() -> None:
    http = MagicMock()
    with pytest.raises(MsGraphMailMessageChanged, match=_CHANGED_ERROR):
        _reader(http).read_message_content(message=_valid_removed_change(), max_chars=10_000)
    http.get.assert_not_called()


def test_validate_message_content_returns_new_instance() -> None:
    original = _valid_message_content(
        subject=_SECRET_SUBJECT,
        unique_body_text="unique",
        from_participant=parse_msgraph_mail_participant(
            _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
        ),
    )
    validated = validate_msgraph_mail_message_content(
        original,
        message=_valid_active_change(),
        max_chars=10_000,
    )
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": _OTHER_MESSAGE_ID},
        {"parent_folder_id": _OTHER_FOLDER_ID},
        {"content_revision": _OTHER_CHANGE_KEY},
        {"mailbox_user_id": _OTHER_MAILBOX_USER_ID},
        {"body_text": 123},
        {"unique_body_text": 123},
        {"reply_to": "bad"},
        {"body_text": None},
        {"remote_id": None},
        {"parent_folder_id": None},
        {"content_revision": None},
        {"to_recipients": []},
        {"from_participant": {"emailAddress": {"address": "bad"}}},
    ],
    ids=[
        "remote_id_mismatch",
        "parent_folder_mismatch",
        "revision_mismatch",
        "mailbox_mismatch",
        "body_text_int",
        "unique_body_int",
        "reply_to_str",
        "missing_body_text",
        "missing_remote_id",
        "missing_parent_folder",
        "missing_revision",
        "recipients_list_not_tuple",
        "malformed_participant",
    ],
)
def test_model_construct_malformed_content_rejected(kwargs: dict[str, object]) -> None:
    base = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _MESSAGE_ID,
        "parent_folder_id": _FOLDER_ID,
        "content_revision": _CHANGE_KEY,
        "body_text": "hello",
        "from_participant": None,
        "sender_participant": None,
        "reply_to": (),
        "to_recipients": (),
        "cc_recipients": (),
        "bcc_recipients": (),
    }
    malformed = MsGraphMailMessageContent.model_construct(**{**base, **kwargs})
    with pytest.raises((ValueError, MsGraphMailMessageChanged), match=f"{_SAFE_ERROR}|{_CHANGED_ERROR}") as exc:
        validate_msgraph_mail_message_content(
            malformed,
            message=_valid_active_change(),
            max_chars=10_000,
        )
    if isinstance(exc.value, ValueError):
        assert exc.value.__cause__ is None


def test_model_construct_missing_body_text_rejected() -> None:
    malformed = MsGraphMailMessageContent.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_MESSAGE_ID,
        parent_folder_id=_FOLDER_ID,
        content_revision=_CHANGE_KEY,
        from_participant=None,
        sender_participant=None,
        reply_to=(),
        to_recipients=(),
        cc_recipients=(),
        bcc_recipients=(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_message_content(
            malformed,
            message=_valid_active_change(),
            max_chars=10_000,
        )
    assert exc.value.__cause__ is None


def test_model_construct_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_mail_message_content(
            "not-content",
            message=_valid_active_change(),
            max_chars=10_000,
        )


# --- limits ---


@pytest.mark.parametrize("max_chars", [0, ABSOLUTE_MAIL_CONTENT_MAX_CHARS + 1, True, "1000", None])
def test_invalid_max_chars_rejected_before_http(max_chars: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_message_content(
            message=_valid_active_change(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


def test_combined_content_over_limit_raises_too_large() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_content_payload(
            body_text="a" * 600,
            include_unique_body=True,
            unique_body_text="b" * 500,
        ),
    )
    with pytest.raises(MsGraphMailContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_message_content(message=_valid_active_change(), max_chars=1000)
    assert exc.value.__cause__ is None
    assert _SECRET_BODY not in str(exc.value)


def test_validate_content_enforces_combined_limit() -> None:
    content = _valid_message_content(
        body_text="a" * 600,
        unique_body_text="b" * 500,
    )
    with pytest.raises(MsGraphMailContentTooLarge, match=_TOO_LARGE_ERROR):
        validate_msgraph_mail_message_content(
            content,
            message=_valid_active_change(),
            max_chars=1000,
        )


@pytest.mark.parametrize("max_chars", [0, ABSOLUTE_MAIL_CONTENT_MAX_CHARS + 1, True, "1000", None])
def test_validate_content_rejects_invalid_max_chars(max_chars: object) -> None:
    content = _valid_message_content()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        validate_msgraph_mail_message_content(
            content,
            message=_valid_active_change(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )


# --- request verification ---


def test_request_exact_path_select_and_prefer_header() -> None:
    http, reader = _setup_happy_path()
    reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    assert http.get.call_args.args[0] == _CONTENT_PATH
    params = http.get.call_args.kwargs["params"]
    assert params["$select"] == _SELECT
    assert "$filter" not in params
    assert "$expand" not in params
    assert http.get.call_args.kwargs["headers"] == _TEXT_BODY_HEADERS


def test_request_uses_quoted_mailbox_and_message_id() -> None:
    http, reader = _setup_happy_path()
    reader.read_message_content(message=_valid_active_change(), max_chars=10_000)
    path = http.get.call_args.args[0]
    assert _QUOTED_MAILBOX in path
    assert _QUOTED_MESSAGE_ID in path


def test_status_404_maps_to_dependency_error() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(status_code=404)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        _reader(http).read_message_content(message=_valid_active_change(), max_chars=10_000)


# --- delegation ---


def test_graph_rest_client_delegates_mail_content() -> None:
    http, _ = _setup_happy_path()
    result = _graph_client(http).read_mail_message_content(message=_valid_active_change())
    assert result.body_text == "Hello, world."


def test_collaboration_suite_delegates_mail_content() -> None:
    http, _ = _setup_happy_path()
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    result = suite.read_mail_message_content(message=_valid_active_change())
    assert result.remote_id == _MESSAGE_ID


def test_integration_delegates_mail_content() -> None:
    http, _ = _setup_happy_path()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    result = integration.read_mail_message_content(message=_valid_active_change())
    assert result.content_revision == _CHANGE_KEY


def test_transport_and_reader_share_injected_http_client() -> None:
    http, _ = _setup_happy_path()
    client = _graph_client(http)
    client.read_mail_message_content(message=_valid_active_change())
    assert client._knowledge_transport._http_client is http
    assert client._mail_content_reader._transport._http_client is http


# --- custom client boundary via integration ---


class _CustomSuiteWithoutMailContent(CollaborationSuite):
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


class _CustomGraphMailContentClient(GraphRestClient):
    def __init__(self, content: MsGraphMailMessageContent, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_content = content

    def read_mail_message_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        max_chars: int = DEFAULT_MAIL_CONTENT_MAX_CHARS,
    ) -> MsGraphMailMessageContent:
        return self._custom_content


def test_custom_client_without_mail_content_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutMailContent(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        integration.read_mail_message_content(message=_valid_active_change())


def test_custom_client_malformed_content_rejected() -> None:
    malformed = MsGraphMailMessageContent.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_MESSAGE_ID,
        parent_folder_id=_FOLDER_ID,
        content_revision=_CHANGE_KEY,
        body_text=123,
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailContentClient(content=malformed, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_mail_message_content(message=_valid_active_change())
    assert exc.value.__cause__ is None


def test_custom_client_valid_content_revalidated() -> None:
    supplied = _valid_message_content(
        subject=_SECRET_SUBJECT,
        body_text="x" * 16_001,
        unique_body_text="unique portion",
        from_participant=parse_msgraph_mail_participant(
            _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
        ),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailContentClient(content=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_mail_message_content(
        message=_valid_active_change(),
        max_chars=20_000,
    )
    assert returned == supplied
    assert returned is not supplied
    assert returned.from_participant is not supplied.from_participant
    assert len(returned.body_text) == 16_001


@pytest.mark.parametrize("max_chars", [0, ABSOLUTE_MAIL_CONTENT_MAX_CHARS + 1, True, "1000", None])
def test_custom_client_invalid_max_chars_rejected(max_chars: object) -> None:
    supplied = _valid_message_content()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailContentClient(content=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_mail_message_content(
            message=_valid_active_change(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )


def test_custom_client_identity_mismatch_rejected() -> None:
    supplied = _valid_message_content(remote_id=_OTHER_MESSAGE_ID)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailContentClient(content=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(MsGraphMailMessageChanged, match=_CHANGED_ERROR):
        integration.read_mail_message_content(message=_valid_active_change())


# --- security ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    content = _valid_message_content(
        subject=_SECRET_SUBJECT,
        body_text=_SECRET_BODY,
        unique_body_text="unique secret",
        content_revision=_CHANGE_KEY,
        internet_message_id=_INTERNET_MESSAGE_ID,
        from_participant=parse_msgraph_mail_participant(
            _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
        ),
    )
    rendered = repr(content)
    assert _SECRET_SUBJECT not in rendered
    assert _SECRET_BODY not in rendered
    assert "unique secret" not in rendered
    assert _CHANGE_KEY not in rendered
    assert _INTERNET_MESSAGE_ID not in rendered
    assert _SECRET_ADDRESS not in rendered
    assert _SECRET_DISPLAY_NAME not in rendered

    participant = parse_msgraph_mail_participant(
        _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    participant_rendered = repr(participant)
    assert _SECRET_ADDRESS not in participant_rendered
    assert _SECRET_DISPLAY_NAME not in participant_rendered

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_message_content(
            message=_valid_active_change(),
            max_chars=0,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert _CHANGE_KEY not in str(exc.value)
