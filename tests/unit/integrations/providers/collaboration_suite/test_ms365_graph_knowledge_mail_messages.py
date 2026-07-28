# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Mail knowledge-read messages delta surface."""

from __future__ import annotations

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
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeSyncResetRequired,
    MsGraphKnowledgeTransport,
    MsGraphMailFolderPage,
    MsGraphMailImportance,
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    MsGraphMailMessageDeltaPage,
    MsGraphMailMessagesReader,
    parse_msgraph_mail_message_change,
    validate_msgraph_mail_message_change,
    validate_msgraph_mail_message_delta_page,
    validate_msgraph_mail_messages_delta_continuation,
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
_CONVERSATION_ID = "conv-abc"
_INTERNET_MESSAGE_ID = "<msg-id@example.com>"
_SUBJECT = "Quarterly report"
_SECRET_SUBJECT = "secret-subject-value"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_FOLDER_ID = quote(_FOLDER_ID, safe="")
_QUOTED_OTHER_FOLDER_ID = quote(_OTHER_FOLDER_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_SECRET_DELTA_TOKEN = "secret-deltatoken-value"
_DELTA_PATH = f"/users/{_QUOTED_MAILBOX}/mailFolders/{_QUOTED_FOLDER_ID}/messages/delta"
_SELECT = (
    "id,parentFolderId,changeKey,conversationId,internetMessageId,subject,"
    "createdDateTime,lastModifiedDateTime,receivedDateTime,sentDateTime,"
    "isRead,isDraft,hasAttachments,importance"
)
_SAFE_ERROR = "unexpected Microsoft Graph mail messages response"
_REQUEST_ERROR = "invalid Microsoft Graph mail messages request"
_CONT_ERROR = "invalid Microsoft Graph mail messages delta continuation"
_VALIDATION_ERROR = "Microsoft Graph Mail messages delta validation is not configured"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Mail messages delta capability"
)
_IMMUTABLE_HEADER = {"Prefer": 'IdType="ImmutableId"'}


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


def _next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_FOLDER_ID}/messages/delta?$skiptoken={_SECRET_TOKEN}"
    )


def _delta_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_FOLDER_ID}/messages/delta?$deltatoken={_SECRET_DELTA_TOKEN}"
    )


def _odata_next_link(folder_literal: str = _FOLDER_ID) -> str:
    escaped = folder_literal.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
        f"mailFolders('{escaped}')/messages/delta?$skiptoken={_SECRET_TOKEN}"
    )


def _page_payload(
    *,
    value: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
    delta_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": [] if value is None else value}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    if delta_link is not None:
        payload["@odata.deltaLink"] = delta_link
    return payload


def _active_message_payload(
    *,
    message_id: str = _MESSAGE_ID,
    parent_folder_id: str = _FOLDER_ID,
    change_key: str = _CHANGE_KEY,
    conversation_id: str | None = None,
    internet_message_id: str | None = None,
    subject: str | None = None,
    created_at: str | None = None,
    received_at: str | None = None,
    sent_at: str | None = None,
    is_read: bool = False,
    is_draft: bool = False,
    has_attachments: bool = False,
    importance: str = "normal",
    include_subject_key: bool = False,
    subject_null: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": message_id,
        "parentFolderId": parent_folder_id,
        "changeKey": change_key,
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
        "isRead": is_read,
        "isDraft": is_draft,
        "hasAttachments": has_attachments,
        "importance": importance,
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    if internet_message_id is not None:
        payload["internetMessageId"] = internet_message_id
    if include_subject_key:
        payload["subject"] = None if subject_null else (subject if subject is not None else "")
    elif subject is not None:
        payload["subject"] = subject
    if created_at is not None:
        payload["createdDateTime"] = created_at
    if received_at is not None:
        payload["receivedDateTime"] = received_at
    if sent_at is not None:
        payload["sentDateTime"] = sent_at
    return payload


def _removed_message_payload(
    *,
    message_id: str = _MESSAGE_ID,
    reason: str = "deleted",
) -> dict[str, Any]:
    return {"id": message_id, "@removed": {"reason": reason}}


def _reader(http: MagicMock) -> MsGraphMailMessagesReader:
    return MsGraphMailMessagesReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


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


def _valid_delta_page(**overrides: object) -> MsGraphMailMessageDeltaPage:
    defaults: dict[str, object] = {
        "items": (_valid_active_change(),),
        "continuation": MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_delta_link(),
        ),
    }
    defaults.update(overrides)
    return MsGraphMailMessageDeltaPage(**defaults)


def _parse_active(payload: dict[str, Any]) -> MsGraphMailMessageChange:
    return parse_msgraph_mail_message_change(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_folder_id=_FOLDER_ID,
    )


# --- parser: active messages ---


def test_parse_active_message_with_required_metadata() -> None:
    change = _parse_active(
        _active_message_payload(
            conversation_id=_CONVERSATION_ID,
            internet_message_id=_INTERNET_MESSAGE_ID,
            subject=_SUBJECT,
            created_at="2024-06-01T10:00:00Z",
            received_at="2024-06-01T11:00:00Z",
            sent_at="2024-06-01T11:30:00Z",
        )
    )
    assert change.kind is MsGraphMailMessageChangeKind.ACTIVE
    assert change.remote_id == _MESSAGE_ID
    assert change.parent_folder_id == _FOLDER_ID
    assert change.change_key == _CHANGE_KEY
    assert change.conversation_id == _CONVERSATION_ID
    assert change.internet_message_id == _INTERNET_MESSAGE_ID
    assert change.subject == _SUBJECT
    assert change.importance is MsGraphMailImportance.NORMAL
    assert change.is_removed is False


def test_parse_active_message_empty_subject() -> None:
    change = _parse_active(_active_message_payload(subject=""))
    assert change.subject == ""


def test_parse_active_message_null_subject() -> None:
    change = _parse_active(
        _active_message_payload(include_subject_key=True, subject_null=True)
    )
    assert change.subject is None


def test_parse_active_message_unknown_importance() -> None:
    change = _parse_active(_active_message_payload(importance="urgent"))
    assert change.importance is MsGraphMailImportance.UNKNOWN


def test_parse_active_message_draft_with_attachments() -> None:
    change = _parse_active(
        _active_message_payload(is_draft=True, has_attachments=True, importance="high")
    )
    assert change.is_draft is True
    assert change.has_attachments is True
    assert change.importance is MsGraphMailImportance.HIGH


def test_parse_active_message_immutable_opaque_id() -> None:
    opaque_id = "AAMkAGI2THSAAA-immutable-opaque-id"
    change = _parse_active(_active_message_payload(message_id=opaque_id))
    assert change.remote_id == opaque_id


# --- parser: removed entries ---


def test_parse_removed_entry_minimal() -> None:
    change = _parse_active(_removed_message_payload())
    assert change.kind is MsGraphMailMessageChangeKind.REMOVED
    assert change.is_removed is True
    assert change.removed_reason == "deleted"
    assert change.parent_folder_id is None
    assert change.change_key is None
    assert change.subject is None


def test_parse_removed_entry_unknown_future_reason() -> None:
    change = _parse_active(_removed_message_payload(reason="futureReason"))
    assert change.removed_reason == "futureReason"


def test_parse_removed_entry_folder_scoped_not_global_deletion() -> None:
    change = _parse_active(_removed_message_payload(reason="deleted"))
    assert change.kind is MsGraphMailMessageChangeKind.REMOVED
    assert change.removed_reason == "deleted"
    assert change.parent_folder_id is None


# --- parser: malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"parentFolderId": _FOLDER_ID},
        {"id": None},
        {"id": ""},
        {"id": 123},
        {"id": _MESSAGE_ID, "@removed": None},
        {"id": _MESSAGE_ID, "@removed": "deleted"},
        {"id": _MESSAGE_ID, "@removed": {}},
        {"id": _MESSAGE_ID, "@removed": {"reason": ""}},
        {"id": _MESSAGE_ID, "@removed": {"reason": 1}},
        _active_message_payload() | {"parentFolderId": _OTHER_FOLDER_ID},
        {"id": _MESSAGE_ID, "parentFolderId": _FOLDER_ID},
        {"id": _MESSAGE_ID, "parentFolderId": _FOLDER_ID, "changeKey": ""},
        {"id": _MESSAGE_ID, "parentFolderId": _FOLDER_ID, "changeKey": 1},
        _active_message_payload() | {"lastModifiedDateTime": "2024-06-01T12:00:00"},
        _active_message_payload() | {"isRead": 1},
        _active_message_payload() | {"isDraft": 0},
        _active_message_payload() | {"hasAttachments": 1},
        _active_message_payload() | {"importance": ""},
        _active_message_payload() | {"importance": 1},
        _active_message_payload(conversation_id=1),
        _active_message_payload(internet_message_id=1),
        _active_message_payload(subject=1),
        _active_message_payload(created_at=123),
    ],
)
def test_parse_malformed_provider_payload(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_mail_message_change(
            payload,
            expected_mailbox_user_id=_MAILBOX_USER_ID,
            expected_folder_id=_FOLDER_ID,
        )
    assert exc.value.__cause__ is None
    assert _MESSAGE_ID not in str(exc.value)
    assert _SUBJECT not in str(exc.value)
    assert _INTERNET_MESSAGE_ID not in str(exc.value)


# --- model and deep validation ---


def test_validate_change_returns_new_instance() -> None:
    original = _valid_active_change(subject=_SECRET_SUBJECT)
    validated = validate_msgraph_mail_message_change(original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": None},
        {"kind": "active"},
        {"scope_folder_id": _OTHER_FOLDER_ID},
        {"change_key": None},
        {"last_modified_at": None},
        {"parent_folder_id": _OTHER_FOLDER_ID},
        {"removed_reason": None, "kind": MsGraphMailMessageChangeKind.REMOVED},
        {"is_read": 1},
        {"last_modified_at": datetime(2024, 6, 1, 12, 0)},
        {"importance": "normal"},
    ],
)
def test_model_construct_malformed_change_rejected(kwargs: dict[str, object]) -> None:
    if kwargs.get("kind") == MsGraphMailMessageChangeKind.REMOVED:
        base = {
            "mailbox_user_id": _MAILBOX_USER_ID,
            "scope_folder_id": _FOLDER_ID,
            "remote_id": _MESSAGE_ID,
            "kind": MsGraphMailMessageChangeKind.REMOVED,
            "removed_reason": "deleted",
        }
    else:
        base = _valid_active_change_kwargs()
    malformed = MsGraphMailMessageChange.model_construct(**{**base, **kwargs})
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_message_change(malformed)
    assert exc.value.__cause__ is None


def test_validate_delta_page_returns_new_instances() -> None:
    original = _valid_delta_page()
    validated = validate_msgraph_mail_message_delta_page(
        original,
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]
    assert validated.continuation is not original.continuation


@pytest.mark.parametrize(
    "page_kwargs",
    [
        {"items": None},
        {"nested_malformed": True},
        {"continuation": "bad"},
    ],
)
def test_model_construct_malformed_page_rejected(page_kwargs: dict[str, object]) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    if page_kwargs.get("items") is None:
        malformed = MsGraphMailMessageDeltaPage.model_construct(continuation=continuation)
    elif page_kwargs.get("nested_malformed"):
        malformed = MsGraphMailMessageDeltaPage.model_construct(
            items=(MsGraphMailMessageChange.model_construct(),),
            continuation=continuation,
        )
    else:
        malformed = MsGraphMailMessageDeltaPage.model_construct(
            items=(_valid_active_change(),),
            continuation=page_kwargs["continuation"],
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_message_delta_page(
            malformed,
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_delta_page_model_rejects_duplicate_remote_ids() -> None:
    item = _valid_active_change()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphMailMessageDeltaPage(items=(item, item), continuation=continuation)


# --- requests and headers ---


def test_initial_request_path_and_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=None,
        limit=25,
    )
    assert http.get.call_args.args[0] == _DELTA_PATH
    params = http.get.call_args.kwargs["params"]
    assert params["$top"] == 25
    assert params["$select"] == _SELECT
    assert "$filter" not in params
    assert "$orderby" not in params
    assert "$expand" not in params
    assert "$search" not in params
    assert "changeType" not in params
    assert http.get.call_args.kwargs["headers"] == _IMMUTABLE_HEADER


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=continuation,
        limit=100,
    )
    assert http.get.call_args.args[0] == _next_link()
    assert http.get.call_args.kwargs.get("params") is None
    assert http.get.call_args.kwargs["headers"] == _IMMUTABLE_HEADER


def test_delta_round_uses_immutable_header() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=continuation,
        limit=100,
    )
    assert http.get.call_args.kwargs["headers"] == _IMMUTABLE_HEADER


@pytest.mark.parametrize("limit", [0, 201, True, "25"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


# --- continuation validation ---


def test_validate_continuation_accepts_next_page_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    validated = validate_msgraph_mail_messages_delta_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_delta_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    validated = validate_msgraph_mail_messages_delta_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.DELTA
    assert validated.url == continuation.url


def test_validate_continuation_accepts_odata_key_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_next_link(),
    )
    validated = validate_msgraph_mail_messages_delta_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_resource_name_case_variations() -> None:
    url = (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/MAILFOLDERS/"
        f"{_QUOTED_FOLDER_ID}/MESSAGES/DELTA?$skiptoken={_SECRET_TOKEN}"
    )
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    validated = validate_msgraph_mail_messages_delta_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_folder_literal_with_escaped_quotes() -> None:
    folder_id = "folder'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_odata_next_link(folder_id),
    )
    validated = validate_msgraph_mail_messages_delta_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=folder_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.DELTA
    assert validated.url == continuation.url


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX_USER_ID, safe='')}/"
        f"mailFolders/{_QUOTED_FOLDER_ID}/messages/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_OTHER_FOLDER_ID}/messages/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_FOLDER_ID}/messages?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_FOLDER_ID}/childFolders?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendar/events?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_FOLDER_ID}/messages/delta/extra?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders('unterminated"
        f"?$skiptoken={_SECRET_TOKEN}",
    ],
)
def test_validate_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_messages_delta_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)


def test_validate_continuation_rejects_wrong_object_type() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_mail_messages_delta_continuation(
            "bad",
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )


def _assert_malformed_continuation_rejected(continuation: object) -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_messages_delta_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _SECRET_DELTA_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert _FOLDER_ID not in str(exc.value)


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_delta_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="delta",
            url=_delta_link(),
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=123,
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url="",
        ),
    ],
)
def test_validate_continuation_rejects_model_construct_malformed(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    _assert_malformed_continuation_rejected(continuation)


def test_validate_delta_page_rejects_malformed_continuation_missing_url() -> None:
    malformed = MsGraphMailMessageDeltaPage.model_construct(
        items=(_valid_active_change(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_message_delta_page(
            malformed,
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_validate_delta_page_rejects_malformed_continuation_missing_kind() -> None:
    malformed = MsGraphMailMessageDeltaPage.model_construct(
        items=(_valid_active_change(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            url=_delta_link(),
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_message_delta_page(
            malformed,
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


# --- delta semantics ---


def test_first_page_with_next_page() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_active_message_payload()],
            next_link=_next_link(),
        )
    )
    page = _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=None,
        limit=50,
    )
    assert page.has_more is True
    assert page.is_complete is False
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_final_page_with_delta() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_active_message_payload()],
            delta_link=_delta_link(),
        )
    )
    page = _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=None,
        limit=50,
    )
    assert page.has_more is False
    assert page.is_complete is True


def test_incremental_round_from_delta_link() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_active_message_payload(message_id=_OTHER_MESSAGE_ID)],
            delta_link=_delta_link(),
        )
    )
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    page = _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=continuation,
        limit=50,
    )
    assert page.items[0].remote_id == _OTHER_MESSAGE_ID


def test_duplicate_id_last_occurrence_wins() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[
                _active_message_payload(message_id=_MESSAGE_ID, subject="Version 1"),
                _active_message_payload(message_id=_OTHER_MESSAGE_ID),
                _removed_message_payload(message_id=_MESSAGE_ID),
            ],
            delta_link=_delta_link(),
        )
    )
    page = _reader(http).read_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
        continuation=None,
        limit=50,
    )
    assert [item.remote_id for item in page.items] == [_OTHER_MESSAGE_ID, _MESSAGE_ID]
    assert page.items[1].kind is MsGraphMailMessageChangeKind.REMOVED


def test_delta_page_requires_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[_active_message_payload()]))
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        _reader(http).read_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            continuation=None,
            limit=50,
        )


def test_status_410_propagates_sync_reset() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(status_code=410)
    with pytest.raises(MsGraphKnowledgeSyncResetRequired, match="must restart"):
        _reader(http).read_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
            continuation=None,
            limit=50,
        )


# --- delegation ---


def test_graph_rest_client_delegates_mail_messages_delta() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(value=[_active_message_payload()], delta_link=_delta_link())
    )
    page = _graph_client(http).read_mail_messages_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
    )
    assert page.items[0].remote_id == _MESSAGE_ID


def test_collaboration_suite_delegates_mail_messages_delta() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_mail_messages_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
    )
    assert page.is_complete is True


def test_integration_delegates_mail_messages_delta() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_mail_messages_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
    )
    assert page.is_complete is True


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    client = _graph_client(http)
    client.read_mail_messages_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
    )
    assert client._knowledge_transport._http_client is http
    assert client._mail_messages_reader._transport._http_client is http


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    client = _graph_client(http)
    client.read_mail_messages_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
    )
    assert client._http_client is http


def test_mail_folders_still_work() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload={"value": []})
    client = _graph_client(http)
    page = client.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_legacy_list_messages_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload={"value": []})
    client = _graph_client(http)
    result = client.list_messages(_MAILBOX_USER_ID, folder="inbox", limit=5)
    assert result.messages == []


def test_legacy_get_message_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={
            "id": "msg-1",
            "subject": "Hello",
            "bodyPreview": "Preview",
            "from": None,
            "receivedDateTime": "2026-01-01T00:00:00Z",
        }
    )
    client = _graph_client(http)
    message = client.get_message(_MAILBOX_USER_ID, "msg-1")
    assert message.id == "msg-1"


def test_drive_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={
            "value": [],
            "@odata.deltaLink": (
                f"https://graph.microsoft.com/v1.0/drives/{quote('drive-1', safe='')}/root/delta?"
                "$deltatoken=tok"
            ),
        }
    )
    client = _graph_client(http)
    page = client.read_drive_delta_page(drive_id="drive-1", limit=10)
    assert page.is_complete is True


class _CustomSuiteWithoutMailMessages(CollaborationSuite):
    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None = None,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ):
        return MsGraphMailFolderPage.model_construct()

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


def test_custom_client_without_mail_messages_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutMailMessages(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        integration.read_mail_messages_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
        )


class _CustomMailMessagesSuite(CollaborationSuite):
    def __init__(self, page: MsGraphMailMessageDeltaPage) -> None:
        self._page = page

    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailMessageDeltaPage:
        return self._page

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


class _CustomGraphMailMessagesClient(GraphRestClient):
    def __init__(self, page: MsGraphMailMessageDeltaPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailMessageDeltaPage:
        return self._custom_page


def test_custom_client_malformed_page_rejected() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailMessagesClient(
                page=MsGraphMailMessageDeltaPage.model_construct(),
                http=MagicMock(),
            )
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_mail_messages_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
        )
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = _valid_delta_page()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailMessagesClient(page=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_mail_messages_delta_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        folder_id=_FOLDER_ID,
    )
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]
    assert returned.continuation is not supplied.continuation


def test_integration_rejects_malformed_continuation_without_url() -> None:
    malformed_page = MsGraphMailMessageDeltaPage(
        items=(_valid_active_change(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
        ),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailMessagesClient(page=malformed_page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        integration.read_mail_messages_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
        )
    assert exc.value.__cause__ is None


def test_custom_client_cross_folder_continuation_rejected() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
            f"{_QUOTED_OTHER_FOLDER_ID}/messages/delta?$deltatoken={_SECRET_DELTA_TOKEN}"
        ),
    )
    page = MsGraphMailMessageDeltaPage(
        items=(_valid_active_change(),),
        continuation=wrong_continuation,
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailMessagesClient(page=page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        integration.read_mail_messages_delta_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            folder_id=_FOLDER_ID,
        )


def test_custom_client_validation_not_configured() -> None:
    page = _valid_delta_page()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomMailMessagesSuite(page=page),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        integration._graph_base_url_for_mail_messages_validation()


# --- security ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    change = _valid_active_change(
        subject=_SECRET_SUBJECT,
        internet_message_id=_INTERNET_MESSAGE_ID,
        change_key=_CHANGE_KEY,
    )
    rendered = repr(change)
    assert _SECRET_SUBJECT not in rendered
    assert _INTERNET_MESSAGE_ID not in rendered
    assert _CHANGE_KEY not in rendered

    page = _valid_delta_page(items=(change,))
    page_rendered = repr(page)
    assert _SECRET_DELTA_TOKEN not in page_rendered
    assert "deltaLink" not in page_rendered
    assert "deltatoken" not in page_rendered

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_delta_page(
            mailbox_user_id="",
            folder_id=_FOLDER_ID,
            continuation=None,
            limit=100,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)
