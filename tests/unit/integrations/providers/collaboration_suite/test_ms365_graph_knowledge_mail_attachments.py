# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Mail knowledge-read attachments surface."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
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
    ABSOLUTE_MAIL_ATTACHMENT_MAX_BYTES,
    DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    MsGraphMailAttachment,
    MsGraphMailAttachmentKind,
    MsGraphMailAttachmentPage,
    MsGraphMailAttachmentTooLarge,
    MsGraphMailAttachmentsReader,
    MsGraphMailFileAttachmentContent,
    MsGraphMailImportance,
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    MsGraphMailMessageChanged,
    parse_msgraph_mail_attachment,
    validate_msgraph_mail_attachment,
    validate_msgraph_mail_attachment_page,
    validate_msgraph_mail_attachments_continuation,
    validate_msgraph_mail_file_attachment_content,
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
_ATTACHMENT_ID = "att-file-001"
_OTHER_ATTACHMENT_ID = "att-other-002"
_ATTACHMENT_NAME = "report.pdf"
_SECRET_ATTACHMENT_NAME = "secret-attachment-name"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_MESSAGE_ID = quote(_MESSAGE_ID, safe="")
_QUOTED_OTHER_MESSAGE_ID = quote(_OTHER_MESSAGE_ID, safe="")
_QUOTED_ATTACHMENT_ID = quote(_ATTACHMENT_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_ATTACHMENTS_PATH = f"/users/{_QUOTED_MAILBOX}/messages/{_QUOTED_MESSAGE_ID}/attachments"
_VALUE_PATH = (
    f"/users/{_QUOTED_MAILBOX}/messages/{_QUOTED_MESSAGE_ID}"
    f"/attachments/{_QUOTED_ATTACHMENT_ID}/$value"
)
_OBSERVATION_PATH = f"/users/{_QUOTED_MAILBOX}/messages/{_QUOTED_MESSAGE_ID}"
_SELECT = "id,name,contentType,size,isInline,contentId,lastModifiedDateTime"
_OBSERVATION_SELECT = (
    "id,parentFolderId,changeKey,lastModifiedDateTime,isRead,isDraft,hasAttachments,importance"
)
_SAFE_ERROR = "unexpected Microsoft Graph Mail attachments response"
_REQUEST_ERROR = "invalid Microsoft Graph Mail attachments request"
_CONT_ERROR = "invalid Microsoft Graph Mail attachments continuation"
_UNSUPPORTED_ERROR = (
    "Microsoft Graph Mail attachment content is not supported for this attachment type"
)
_INVALID_RESPONSE = "Microsoft Graph Mail attachment response is invalid"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Mail attachments capability"
)
_VALIDATION_ERROR = "Microsoft Graph Mail attachment validation is not configured"
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


def _attachments_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
        f"{_QUOTED_MESSAGE_ID}/attachments?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_attachments_next_link(message_id: str = _MESSAGE_ID) -> str:
    escaped = message_id.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
        f"messages('{escaped}')/attachments?$skiptoken={_SECRET_TOKEN}"
    )


def _page_payload(
    *,
    value: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": [] if value is None else value}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    return payload


def _observation_payload(
    *,
    message_id: str = _MESSAGE_ID,
    parent_folder_id: str = _FOLDER_ID,
    change_key: str = _CHANGE_KEY,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "parentFolderId": parent_folder_id,
        "changeKey": change_key,
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
        "isRead": False,
        "isDraft": False,
        "hasAttachments": True,
        "importance": "normal",
    }


def _attachment_payload(
    *,
    attachment_id: str = _ATTACHMENT_ID,
    odata_type: str = "#microsoft.graph.fileAttachment",
    name: str = _ATTACHMENT_NAME,
    content_type: str | None = "application/pdf",
    size: int = 42,
    is_inline: bool = False,
    content_id: str | None = None,
    include_content_type_key: bool = True,
    content_type_null: bool = False,
    include_content_id_key: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "@odata.type": odata_type,
        "id": attachment_id,
        "name": name,
        "size": size,
        "isInline": is_inline,
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
    }
    if include_content_type_key:
        payload["contentType"] = None if content_type_null else content_type
    if include_content_id_key:
        payload["contentId"] = content_id
    return payload


class _FakeStreamContext:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (b"hello",),
        iter_raises: Exception | None = None,
    ) -> None:
        self._status_code = status_code
        self._headers = {} if headers is None else headers
        self._chunks = chunks
        self._iter_raises = iter_raises

    def __enter__(self) -> MagicMock:
        response = MagicMock()
        response.status_code = self._status_code
        response.headers = self._headers

        def _iter_bytes() -> Iterator[bytes]:
            if self._iter_raises is not None:
                raise self._iter_raises
            yield from self._chunks

        response.iter_bytes = _iter_bytes
        return response

    def __exit__(self, *args: object) -> None:
        return None


class _BrokenHeaderMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Iterator[tuple[str, str]]:
        raise RuntimeError("broken headers")


def _valid_active_change(**overrides: object) -> MsGraphMailMessageChange:
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
        "has_attachments": True,
        "importance": MsGraphMailImportance.NORMAL,
    }
    defaults.update(overrides)
    return MsGraphMailMessageChange(**defaults)


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


def _valid_attachment_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "message_remote_id": _MESSAGE_ID,
        "scope_folder_id": _FOLDER_ID,
        "message_revision": _CHANGE_KEY,
        "remote_id": _ATTACHMENT_ID,
        "kind": MsGraphMailAttachmentKind.FILE,
        "name": _ATTACHMENT_NAME,
        "content_type": "application/pdf",
        "size_bytes": 42,
        "is_inline": False,
        "content_id": None,
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return defaults


def _valid_attachment(**overrides: object) -> MsGraphMailAttachment:
    return MsGraphMailAttachment(**_valid_attachment_kwargs(**overrides))


def _valid_attachment_page(**overrides: object) -> MsGraphMailAttachmentPage:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "message_remote_id": _MESSAGE_ID,
        "scope_folder_id": _FOLDER_ID,
        "message_revision": _CHANGE_KEY,
        "items": (_valid_attachment(),),
        "continuation": None,
    }
    defaults.update(overrides)
    return MsGraphMailAttachmentPage(**defaults)


def _valid_file_content(
    data: bytes = b"hello-world",
    **overrides: object,
) -> MsGraphMailFileAttachmentContent:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "message_remote_id": _MESSAGE_ID,
        "scope_folder_id": _FOLDER_ID,
        "message_revision": _CHANGE_KEY,
        "attachment_remote_id": _ATTACHMENT_ID,
        "name": _ATTACHMENT_NAME,
        "content_type": "application/pdf",
        "is_inline": False,
        "content_id": None,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }
    defaults.update(overrides)
    return MsGraphMailFileAttachmentContent(**defaults)


def _reader(http: MagicMock) -> MsGraphMailAttachmentsReader:
    return MsGraphMailAttachmentsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
        graph_http_client=http,
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_attachment(payload: dict[str, Any]) -> MsGraphMailAttachment:
    return parse_msgraph_mail_attachment(payload, message=_valid_active_change())


def _setup_attachments_page(
    http: MagicMock,
    *,
    attachments: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> None:
    observation = _observation_payload()
    attachments_payload = _page_payload(
        value=attachments if attachments is not None else [_attachment_payload()],
        next_link=next_link,
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=attachments_payload),
        _json_response(payload=observation),
    ]


def _setup_file_content(
    http: MagicMock,
    *,
    file_bytes: bytes = b"hello-world",
    attachment: MsGraphMailAttachment | None = None,
) -> MsGraphMailAttachment:
    if attachment is None:
        attachment = _valid_attachment(size_bytes=len(file_bytes))
    observation = _observation_payload()
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=observation),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(file_bytes))},
        chunks=(file_bytes,),
    )
    return attachment


# --- parser: attachment kinds ---


@pytest.mark.parametrize(
    ("odata_type", "kind", "binary_supported"),
    [
        ("#microsoft.graph.fileAttachment", MsGraphMailAttachmentKind.FILE, True),
        ("#microsoft.graph.itemAttachment", MsGraphMailAttachmentKind.ITEM, False),
        ("#microsoft.graph.referenceAttachment", MsGraphMailAttachmentKind.REFERENCE, False),
        ("#microsoft.graph.unknownFutureType", MsGraphMailAttachmentKind.UNKNOWN, False),
    ],
)
def test_parse_attachment_kinds_and_binary_content_supported(
    odata_type: str,
    kind: MsGraphMailAttachmentKind,
    binary_supported: bool,
) -> None:
    attachment = _parse_attachment(_attachment_payload(odata_type=odata_type))
    assert attachment.kind is kind
    assert attachment.binary_content_supported is binary_supported


def test_parse_file_attachment_with_metadata() -> None:
    attachment = _parse_attachment(
        _attachment_payload(
            include_content_id_key=True,
            content_id="cid-inline-1",
            is_inline=True,
            content_type="image/png",
        )
    )
    assert attachment.remote_id == _ATTACHMENT_ID
    assert attachment.name == _ATTACHMENT_NAME
    assert attachment.content_type == "image/png"
    assert attachment.size_bytes == 42
    assert attachment.is_inline is True
    assert attachment.content_id == "cid-inline-1"
    assert attachment.message_revision == _CHANGE_KEY
    assert attachment.scope_folder_id == _FOLDER_ID


def test_parse_attachment_null_content_type() -> None:
    attachment = _parse_attachment(
        _attachment_payload(include_content_type_key=True, content_type_null=True)
    )
    assert attachment.content_type is None


def test_parse_attachment_missing_optional_content_type_key() -> None:
    payload = _attachment_payload(include_content_type_key=False)
    attachment = _parse_attachment(payload)
    assert attachment.content_type is None


@pytest.mark.parametrize("content_type", ["", "   "])
def test_parse_attachment_rejects_empty_content_type(content_type: str) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _parse_attachment(_attachment_payload(content_type=content_type))
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("content_id", ["", "   "])
def test_parse_attachment_rejects_empty_content_id(content_id: str) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _parse_attachment(
            _attachment_payload(include_content_id_key=True, content_id=content_id)
        )
    assert exc.value.__cause__ is None


def test_parse_attachment_removed_message_rejected() -> None:
    with pytest.raises(MsGraphMailMessageChanged):
        parse_msgraph_mail_attachment(
            _attachment_payload(),
            message=_valid_removed_change(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"@odata.type": "#microsoft.graph.fileAttachment"},
        {"@odata.type": "#microsoft.graph.fileAttachment", "id": ""},
        {"@odata.type": "#microsoft.graph.fileAttachment", "id": _ATTACHMENT_ID},
        _attachment_payload() | {"name": 123},
        _attachment_payload() | {"size": -1},
        _attachment_payload() | {"size": "42"},
        _attachment_payload() | {"isInline": 1},
        _attachment_payload() | {"lastModifiedDateTime": "2024-06-01T12:00:00"},
        _attachment_payload() | {"@odata.type": ""},
        _attachment_payload() | {"@odata.type": 123},
        _attachment_payload() | {"contentType": 123},
        _attachment_payload(include_content_type_key=True, content_type=""),
        _attachment_payload(include_content_type_key=True, content_type="   "),
        _attachment_payload(include_content_id_key=True, content_id=123),
        _attachment_payload(include_content_id_key=True, content_id=""),
        _attachment_payload(include_content_id_key=True, content_id="   "),
        _attachment_payload() | {"id": "\x00bad"},
    ],
)
def test_parse_malformed_provider_payload(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_mail_attachment(payload, message=_valid_active_change())
    assert exc.value.__cause__ is None
    assert _ATTACHMENT_NAME not in str(exc.value)
    assert _ATTACHMENT_ID not in str(exc.value)


# --- model and deep validation ---


def test_validate_attachment_returns_new_instance() -> None:
    original = _valid_attachment(name=_SECRET_ATTACHMENT_NAME)
    validated = validate_msgraph_mail_attachment(original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": None},
        {"kind": "file"},
        {"message_revision": None},
        {"size_bytes": -1},
        {"size_bytes": "42"},
        {"is_inline": 1},
        {"last_modified_at": datetime(2024, 6, 1, 12, 0)},
    ],
)
def test_model_construct_malformed_attachment_rejected(kwargs: dict[str, object]) -> None:
    malformed = MsGraphMailAttachment.model_construct(**{**_valid_attachment_kwargs(), **kwargs})
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_attachment(malformed)
    assert exc.value.__cause__ is None


def test_validate_attachment_page_returns_new_instances() -> None:
    original = _valid_attachment_page()
    validated = validate_msgraph_mail_attachment_page(
        original,
        message=_valid_active_change(),
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]


@pytest.mark.parametrize(
    "page_kwargs",
    [
        {"items": None},
        {"nested_malformed": True},
        {"continuation": "bad"},
    ],
)
def test_model_construct_malformed_page_rejected(page_kwargs: dict[str, object]) -> None:
    if page_kwargs.get("items") is None:
        malformed = MsGraphMailAttachmentPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            message_remote_id=_MESSAGE_ID,
            scope_folder_id=_FOLDER_ID,
            message_revision=_CHANGE_KEY,
        )
    elif page_kwargs.get("nested_malformed"):
        malformed = MsGraphMailAttachmentPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            message_remote_id=_MESSAGE_ID,
            scope_folder_id=_FOLDER_ID,
            message_revision=_CHANGE_KEY,
            items=(MsGraphMailAttachment.model_construct(),),
        )
    else:
        malformed = MsGraphMailAttachmentPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            message_remote_id=_MESSAGE_ID,
            scope_folder_id=_FOLDER_ID,
            message_revision=_CHANGE_KEY,
            items=(_valid_attachment(),),
            continuation=page_kwargs["continuation"],
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_attachment_page(
            malformed,
            message=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_attachment_page_model_rejects_duplicate_remote_ids() -> None:
    item = _valid_attachment()
    page = MsGraphMailAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        message_remote_id=_MESSAGE_ID,
        scope_folder_id=_FOLDER_ID,
        message_revision=_CHANGE_KEY,
        items=(item, item),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_mail_attachment_page(
            page,
            message=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_page_rejects_stale_message_revision() -> None:
    page = _valid_attachment_page(message_revision=_OTHER_CHANGE_KEY)
    with pytest.raises(MsGraphMailMessageChanged):
        validate_msgraph_mail_attachment_page(
            page,
            message=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_file_content_returns_new_instance() -> None:
    data = b"content-bytes"
    original = _valid_file_content(data=data)
    validated = validate_msgraph_mail_file_attachment_content(
        original,
        message=_valid_active_change(),
        attachment=_valid_attachment(size_bytes=len(data)),
        max_bytes=1024,
    )
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        ({"data": b"x"}, ValueError),
        ({"size_bytes": 99}, ValueError),
        ({"content_hash": "not-a-valid-sha256-hash"}, ValueError),
        ({"attachment_remote_id": _OTHER_ATTACHMENT_ID}, MsGraphMailMessageChanged),
        ({"message_revision": _OTHER_CHANGE_KEY}, MsGraphMailMessageChanged),
        ({"data": bytearray(b"abc")}, ValueError),
        ({"size_bytes": True}, ValueError),
        ({"name": "other.pdf"}, MsGraphMailMessageChanged),
        ({"content_type": "text/plain"}, MsGraphMailMessageChanged),
        ({"is_inline": True}, MsGraphMailMessageChanged),
        ({"content_id": "other-cid"}, MsGraphMailMessageChanged),
    ],
)
def test_model_construct_malformed_file_content_rejected(
    kwargs: dict[str, object],
    error_type: type[BaseException],
) -> None:
    data = b"abc"
    base = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "message_remote_id": _MESSAGE_ID,
        "scope_folder_id": _FOLDER_ID,
        "message_revision": _CHANGE_KEY,
        "attachment_remote_id": _ATTACHMENT_ID,
        "name": _ATTACHMENT_NAME,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
        "is_inline": False,
    }
    malformed = MsGraphMailFileAttachmentContent.model_construct(**{**base, **kwargs})
    with pytest.raises(error_type, match=_SAFE_ERROR if error_type is ValueError else "changed|supported") as exc:
        validate_msgraph_mail_file_attachment_content(
            malformed,
            message=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=len(data)),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    "missing_field",
    ["name", "is_inline", "data", "size_bytes", "content_hash"],
)
def test_model_construct_missing_file_content_field_rejected(missing_field: str) -> None:
    data = b"abc"
    base: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "message_remote_id": _MESSAGE_ID,
        "scope_folder_id": _FOLDER_ID,
        "message_revision": _CHANGE_KEY,
        "attachment_remote_id": _ATTACHMENT_ID,
        "name": _ATTACHMENT_NAME,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
        "is_inline": False,
    }
    del base[missing_field]
    malformed = MsGraphMailFileAttachmentContent.model_construct(**base)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_file_attachment_content(
            malformed,
            message=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=len(data)),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_validate_file_content_data_length_mismatch_with_attachment_size() -> None:
    data = b"abc"
    content = _valid_file_content(data=data, size_bytes=len(data))
    with pytest.raises(MsGraphMailMessageChanged):
        validate_msgraph_mail_file_attachment_content(
            content,
            message=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=99),
            max_bytes=1024,
        )


def test_validate_file_content_rejects_non_file_attachment() -> None:
    data = b"abc"
    content = _valid_file_content(data=data)
    with pytest.raises(IntegrationConfigurationError, match=_UNSUPPORTED_ERROR):
        validate_msgraph_mail_file_attachment_content(
            content,
            message=_valid_active_change(),
            attachment=_valid_attachment(kind=MsGraphMailAttachmentKind.ITEM, size_bytes=len(data)),
            max_bytes=1024,
        )


def test_validate_file_content_too_large() -> None:
    data = b"toolarge"
    content = _valid_file_content(data=data)
    with pytest.raises(MsGraphMailAttachmentTooLarge):
        validate_msgraph_mail_file_attachment_content(
            content,
            message=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=len(data)),
            max_bytes=3,
        )


# --- requests and headers ---


def test_initial_request_path_and_params() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    _reader(http).read_attachments_page(
        message=_valid_active_change(),
        continuation=None,
        limit=25,
    )
    list_call = http.get.call_args_list[1]
    assert list_call.args[0] == _ATTACHMENTS_PATH
    params = list_call.kwargs["params"]
    assert params["$top"] == 25
    assert params["$select"] == _SELECT
    assert "$filter" not in params
    assert "$orderby" not in params
    assert "$expand" not in params
    assert "$search" not in params
    assert list_call.kwargs["headers"] == _IMMUTABLE_HEADER


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    observation = _observation_payload()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_attachments_next_link(),
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=_page_payload()),
        _json_response(payload=observation),
    ]
    _reader(http).read_attachments_page(
        message=_valid_active_change(),
        continuation=continuation,
        limit=100,
    )
    continuation_call = http.get.call_args_list[1]
    assert continuation_call.args[0] == _attachments_next_link()
    assert continuation_call.kwargs.get("params") is None
    assert continuation_call.kwargs["headers"] == _IMMUTABLE_HEADER


def test_observation_requests_use_select_and_immutable_header() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    _reader(http).read_attachments_page(
        message=_valid_active_change(),
        continuation=None,
        limit=10,
    )
    pre_observation = http.get.call_args_list[0]
    post_observation = http.get.call_args_list[2]
    assert pre_observation.args[0] == _OBSERVATION_PATH
    assert pre_observation.kwargs["params"]["$select"] == _OBSERVATION_SELECT
    assert pre_observation.kwargs["headers"] == _IMMUTABLE_HEADER
    assert post_observation.args[0] == _OBSERVATION_PATH


@pytest.mark.parametrize("limit", [0, 201, True, "25"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_attachments_page(
            message=_valid_active_change(),
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


@pytest.mark.parametrize("max_bytes", [0, ABSOLUTE_MAIL_ATTACHMENT_MAX_BYTES + 1, True, "10", None])
def test_invalid_max_bytes_rejected_before_http(max_bytes: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=_valid_attachment(),
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


# --- continuation validation ---


def test_validate_continuation_accepts_next_page_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_attachments_next_link(),
    )
    validated = validate_msgraph_mail_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        message_id=_MESSAGE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_odata_key_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_attachments_next_link(),
    )
    validated = validate_msgraph_mail_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        message_id=_MESSAGE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_message_literal_with_escaped_quotes() -> None:
    message_id = "msg'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_attachments_next_link(message_id),
    )
    validated = validate_msgraph_mail_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        message_id=message_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_uppercase_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/USERS/{_QUOTED_MAILBOX}/MESSAGES/"
            f"{_QUOTED_MESSAGE_ID}/ATTACHMENTS?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_mail_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        message_id=_MESSAGE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_percent_encoded_message_literal() -> None:
    message_id = "msg/special"
    encoded = quote(message_id, safe="")
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
            f"messages('{encoded}')/attachments?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_mail_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        message_id=message_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX_USER_ID, safe='')}/"
        f"messages/{_QUOTED_MESSAGE_ID}/attachments?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
        f"{_QUOTED_OTHER_MESSAGE_ID}/attachments?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
        f"{_QUOTED_MESSAGE_ID}/attachments/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{quote(_FOLDER_ID, safe='')}/messages/{_QUOTED_MESSAGE_ID}/attachments?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
        f"{_QUOTED_MESSAGE_ID}/attachments/{_QUOTED_ATTACHMENT_ID}/$value",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
        f"{_QUOTED_MESSAGE_ID}/attachments/extra?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/children?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages('unterminated"
        f"?$skiptoken={_SECRET_TOKEN}",
    ],
)
def test_validate_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_attachments_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            message_id=_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)


def test_validate_continuation_rejects_delta_kind() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_attachments_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_mail_attachments_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            message_id=_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_continuation_rejects_wrong_object_type() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_mail_attachments_continuation(
            "bad",
            mailbox_user_id=_MAILBOX_USER_ID,
            message_id=_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )


def _assert_malformed_continuation_rejected(continuation: object) -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_attachments_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            message_id=_MESSAGE_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert _MESSAGE_ID not in str(exc.value)


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_attachments_next_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="next_page",
            url=_attachments_next_link(),
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
def test_validate_continuation_rejects_model_construct_malformed(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    _assert_malformed_continuation_rejected(continuation)


def test_validate_page_rejects_delta_continuation_kind() -> None:
    malformed = MsGraphMailAttachmentPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        message_remote_id=_MESSAGE_ID,
        scope_folder_id=_FOLDER_ID,
        message_revision=_CHANGE_KEY,
        items=(_valid_attachment(),),
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_attachments_next_link(),
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_mail_attachment_page(
            malformed,
            message=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


# --- paging semantics ---


def test_first_page_with_next_page() -> None:
    http = MagicMock()
    _setup_attachments_page(http, next_link=_attachments_next_link())
    page = _reader(http).read_attachments_page(
        message=_valid_active_change(),
        continuation=None,
        limit=50,
    )
    assert page.has_more is True
    assert page.continuation is not None
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_final_page_without_continuation() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    page = _reader(http).read_attachments_page(
        message=_valid_active_change(),
        continuation=None,
        limit=50,
    )
    assert page.has_more is False
    assert page.continuation is None


def test_message_changed_before_list() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_observation_payload(change_key=_OTHER_CHANGE_KEY),
    )
    with pytest.raises(MsGraphMailMessageChanged):
        _reader(http).read_attachments_page(
            message=_valid_active_change(),
            continuation=None,
            limit=50,
        )
    http.get.assert_called_once()


def test_message_changed_after_list() -> None:
    http = MagicMock()
    observation_before = _observation_payload()
    observation_after = _observation_payload(change_key=_OTHER_CHANGE_KEY)
    http.get.side_effect = [
        _json_response(payload=observation_before),
        _json_response(payload=_page_payload(value=[_attachment_payload()])),
        _json_response(payload=observation_after),
    ]
    with pytest.raises(MsGraphMailMessageChanged):
        _reader(http).read_attachments_page(
            message=_valid_active_change(),
            continuation=None,
            limit=50,
        )


# --- unsupported attachment types for content ---


@pytest.mark.parametrize(
    "kind",
    [
        MsGraphMailAttachmentKind.ITEM,
        MsGraphMailAttachmentKind.REFERENCE,
        MsGraphMailAttachmentKind.UNKNOWN,
    ],
)
def test_read_file_content_rejects_non_file_attachment(kind: MsGraphMailAttachmentKind) -> None:
    http = MagicMock()
    attachment = _valid_attachment(kind=kind)
    with pytest.raises(IntegrationConfigurationError, match=_UNSUPPORTED_ERROR):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


def test_read_file_content_attachment_size_exceeds_limit_before_request() -> None:
    http = MagicMock()
    attachment = _valid_attachment(size_bytes=500)
    with pytest.raises(MsGraphMailAttachmentTooLarge):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=100,
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


def test_read_file_content_stale_attachment_revision() -> None:
    http = MagicMock()
    attachment = _valid_attachment(message_revision=_OTHER_CHANGE_KEY)
    with pytest.raises(MsGraphMailMessageChanged):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    http.get.assert_not_called()


# --- file attachment content streaming ---


def test_download_small_file_with_sha256() -> None:
    data = b"small-file-content"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    result = _reader(http).read_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert isinstance(result, MsGraphMailFileAttachmentContent)
    assert result.data == data
    assert result.size_bytes == len(data)
    assert result.content_hash == hashlib.sha256(data).hexdigest()
    assert result.content_type == "application/pdf"
    assert result.message_revision == _CHANGE_KEY


def test_download_empty_file() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"")
    result = _reader(http).read_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert result.data == b""
    assert result.size_bytes == 0


def test_download_multiple_chunks() -> None:
    data = b"hello-world"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(b"hello", b"-world"),
    )
    result = _reader(http).read_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert result.data == data


def test_download_without_content_length() -> None:
    data = b"12345"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    http.stream.return_value = _FakeStreamContext(chunks=(data,))
    result = _reader(http).read_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert result.data == data


def test_download_content_length_exceeds_limit() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "100"},
        chunks=(b"x",),
    )
    with pytest.raises(MsGraphMailAttachmentTooLarge):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=10,
        )


def test_download_malformed_content_length() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "not-a-number"},
        chunks=(b"x",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_bytes_exceed_limit_during_stream() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"abcdefghij")
    http.stream.return_value = _FakeStreamContext(
        chunks=(b"a" * 5, b"b" * 10),
    )
    with pytest.raises(MsGraphMailAttachmentTooLarge):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=10,
        )


def test_download_content_length_mismatch() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"short")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "10"},
        chunks=(b"short",),
    )
    with pytest.raises(MsGraphMailMessageChanged):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_attachment_size_mismatch() -> None:
    data = b"hello"
    http = MagicMock()
    attachment = _valid_attachment(size_bytes=99)
    http.get.side_effect = [
        _json_response(payload=_observation_payload()),
        _json_response(payload=_observation_payload()),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(data,),
    )
    with pytest.raises(MsGraphMailMessageChanged):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_chunk_not_bytes() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")

    class _BadChunkStream(_FakeStreamContext):
        def __enter__(self) -> MagicMock:
            response = super().__enter__()
            response.iter_bytes = lambda: iter(["not-bytes"])  # type: ignore[assignment]
            return response

    http.stream.return_value = _BadChunkStream(chunks=())
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [206, 302, 500])
def test_download_bad_status(status_code: int) -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationDependencyError):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [401, 403])
def test_download_configuration_errors(status_code: int) -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationConfigurationError):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_stream_transport_exception() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.side_effect = RuntimeError("stream failed")
    with pytest.raises(IntegrationDependencyError) as exc:
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_request_path_and_headers() -> None:
    data = b"bytes"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    _reader(http).read_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    call = http.stream.call_args
    assert call.args[1] == _VALUE_PATH
    assert call.kwargs["follow_redirects"] is False
    assert call.kwargs["headers"] == {
        "Accept": "application/octet-stream",
        "Prefer": 'IdType="ImmutableId"',
    }
    assert "params" not in call.kwargs


def test_download_integer_header_key_rejected() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers={1: "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE) as exc:
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_duplicate_content_length_different_case_rejected() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5", "content-length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_headers_items_raises_rejected() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers=_BrokenHeaderMapping(),
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_message_changed_after_download_bytes_not_returned() -> None:
    http = MagicMock()
    data = b"hello"
    attachment = _valid_attachment(size_bytes=len(data))
    http.get.side_effect = [
        _json_response(payload=_observation_payload()),
        _json_response(payload=_observation_payload(change_key=_OTHER_CHANGE_KEY)),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(data,),
    )
    with pytest.raises(MsGraphMailMessageChanged):
        _reader(http).read_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_observation_before_and_after_download() -> None:
    data = b"payload"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    _reader(http).read_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert http.get.call_count == 2
    assert http.get.call_args_list[0].args[0] == _OBSERVATION_PATH
    assert http.get.call_args_list[1].args[0] == _OBSERVATION_PATH
    assert http.stream.call_count == 1


# --- delegation ---


def test_graph_rest_client_delegates_mail_attachments_page() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    page = _graph_client(http).read_mail_attachments_page(message=_valid_active_change())
    assert page.items[0].remote_id == _ATTACHMENT_ID


def test_graph_rest_client_delegates_mail_file_attachment_content() -> None:
    data = b"delegated"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    result = _graph_client(http).read_mail_file_attachment_content(
        message=_valid_active_change(),
        attachment=attachment,
    )
    assert result.data == data


def test_collaboration_suite_delegates_mail_attachments() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_mail_attachments_page(message=_valid_active_change())
    assert page.items[0].kind is MsGraphMailAttachmentKind.FILE


def test_integration_delegates_mail_attachments() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_mail_attachments_page(message=_valid_active_change())
    assert page.has_more is False


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    client = _graph_client(http)
    client.read_mail_attachments_page(message=_valid_active_change())
    assert client._knowledge_transport._http_client is http
    assert client._mail_attachments_reader._transport._http_client is http
    assert client._mail_attachments_reader._graph_http_client is http


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    client = _graph_client(http)
    client.read_mail_attachments_page(message=_valid_active_change())
    assert client._http_client is http


class _CustomSuiteWithoutMailAttachments(CollaborationSuite):
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


def test_custom_client_without_mail_attachments_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutMailAttachments(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        integration.read_mail_attachments_page(message=_valid_active_change())


class _CustomMailAttachmentsSuite(CollaborationSuite):
    def __init__(self, page: MsGraphMailAttachmentPage) -> None:
        self._page = page

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailAttachmentPage:
        return self._page

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int = DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphMailFileAttachmentContent:
        data = b"custom"
        return MsGraphMailFileAttachmentContent(
            mailbox_user_id=message.mailbox_user_id,
            message_remote_id=message.remote_id,
            scope_folder_id=message.scope_folder_id,
            message_revision=message.change_key or "",
            attachment_remote_id=attachment.remote_id,
            name=attachment.name,
            content_type=attachment.content_type,
            is_inline=attachment.is_inline,
            content_id=attachment.content_id,
            data=data,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
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


class _CustomGraphMailAttachmentsClient(GraphRestClient):
    def __init__(self, page: MsGraphMailAttachmentPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailAttachmentPage:
        return self._custom_page


def test_custom_client_malformed_page_rejected() -> None:
    malformed_page = MsGraphMailAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        message_remote_id=_MESSAGE_ID,
        scope_folder_id=_FOLDER_ID,
        message_revision=_CHANGE_KEY,
        items=(MsGraphMailAttachment.model_construct(),),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailAttachmentsClient(page=malformed_page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_mail_attachments_page(message=_valid_active_change())
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = _valid_attachment_page()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailAttachmentsClient(page=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_mail_attachments_page(message=_valid_active_change())
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_integration_rejects_malformed_continuation_without_url() -> None:
    malformed_page = MsGraphMailAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        message_remote_id=_MESSAGE_ID,
        scope_folder_id=_FOLDER_ID,
        message_revision=_CHANGE_KEY,
        items=(_valid_attachment(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailAttachmentsClient(page=malformed_page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_mail_attachments_page(message=_valid_active_change())
    assert exc.value.__cause__ is None


def test_custom_client_cross_message_continuation_rejected() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
            f"{_QUOTED_OTHER_MESSAGE_ID}/attachments?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    page = MsGraphMailAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        message_remote_id=_MESSAGE_ID,
        scope_folder_id=_FOLDER_ID,
        message_revision=_CHANGE_KEY,
        items=(_valid_attachment(),),
        continuation=wrong_continuation,
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphMailAttachmentsClient(page=page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        integration.read_mail_attachments_page(message=_valid_active_change())


def test_custom_client_validation_not_configured() -> None:
    page = _valid_attachment_page()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomMailAttachmentsSuite(page=page),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        integration._graph_base_url_for_mail_attachments_validation()


class _CustomMailFileContentSuite(CollaborationSuite):
    def __init__(self, attachment: MsGraphMailAttachment) -> None:
        self._attachment = attachment

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailAttachmentPage:
        raise NotImplementedError

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int = DEFAULT_MAIL_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphMailFileAttachmentContent:
        data = b"custom-binary"
        return MsGraphMailFileAttachmentContent(
            mailbox_user_id=message.mailbox_user_id,
            message_remote_id=message.remote_id,
            scope_folder_id=message.scope_folder_id,
            message_revision=message.change_key or "",
            attachment_remote_id=attachment.remote_id,
            name=attachment.name,
            content_type=attachment.content_type,
            is_inline=attachment.is_inline,
            content_id=attachment.content_id,
            data=data,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
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
    "kind",
    [
        MsGraphMailAttachmentKind.ITEM,
        MsGraphMailAttachmentKind.REFERENCE,
        MsGraphMailAttachmentKind.UNKNOWN,
    ],
)
def test_custom_client_non_file_attachment_content_rejected(kind: MsGraphMailAttachmentKind) -> None:
    attachment = _valid_attachment(kind=kind, size_bytes=13)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomMailFileContentSuite(attachment=attachment),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_UNSUPPORTED_ERROR):
        integration.read_mail_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
        )


@pytest.mark.parametrize("max_bytes", [0, ABSOLUTE_MAIL_ATTACHMENT_MAX_BYTES + 1, True, "10", None])
def test_custom_client_invalid_max_bytes_rejected(max_bytes: object) -> None:
    attachment = _valid_attachment(size_bytes=13)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomMailFileContentSuite(attachment=attachment),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_mail_file_attachment_content(
            message=_valid_active_change(),
            attachment=attachment,
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )


# --- security ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    attachment = _valid_attachment(name=_SECRET_ATTACHMENT_NAME)
    rendered = repr(attachment)
    assert _SECRET_ATTACHMENT_NAME not in rendered
    assert _CHANGE_KEY not in rendered

    page = _valid_attachment_page(
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_attachments_next_link(),
        )
    )
    page_rendered = repr(page)
    assert _SECRET_TOKEN not in page_rendered
    assert "nextLink" not in page_rendered
    assert "skiptoken" not in page_rendered

    data = b"secret-bytes-payload"
    content = _valid_file_content(data=data)
    content_rendered = repr(content)
    assert data.decode() not in content_rendered


def test_default_max_bytes_constants() -> None:
    assert DEFAULT_MAIL_ATTACHMENT_MAX_BYTES == 10 * 1024 * 1024
    assert ABSOLUTE_MAIL_ATTACHMENT_MAX_BYTES == 25 * 1024 * 1024
