# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Mail knowledge-read folders surface."""

from __future__ import annotations

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
    MsGraphKnowledgeTransport,
    MsGraphMailFolder,
    MsGraphMailFolderPage,
    MsGraphMailFoldersReader,
    parse_msgraph_mail_folder,
    validate_msgraph_mail_folder,
    validate_msgraph_mail_folder_page,
    validate_msgraph_mail_folders_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_FOLDER_ID = "folder-abc-123"
_PARENT_FOLDER_ID = "parent-folder-456"
_OTHER_FOLDER_ID = "other-folder"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_FOLDER_ID = quote(_FOLDER_ID, safe="")
_QUOTED_PARENT_FOLDER_ID = quote(_PARENT_FOLDER_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_DISPLAY_NAME = "Inbox"
_HIDDEN_DISPLAY_NAME = "Hidden Folder"
_ROOT_PATH = f"/users/{_QUOTED_MAILBOX}/mailFolders"
_CHILD_PATH = (
    f"/users/{_QUOTED_MAILBOX}/mailFolders/{_QUOTED_PARENT_FOLDER_ID}/childFolders"
)
_SELECT = (
    "id,displayName,parentFolderId,childFolderCount,totalItemCount,unreadItemCount,isHidden"
)
_SAFE_ERROR = "unexpected Microsoft Graph mail folders response"
_REQUEST_ERROR = "invalid Microsoft Graph mailbox folder request"
_CONT_ERROR = "invalid Microsoft Graph mail folders continuation"


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


def _root_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders"
        f"?$skiptoken={_SECRET_TOKEN}"
    )


def _child_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"{_QUOTED_PARENT_FOLDER_ID}/childFolders?$skiptoken={_SECRET_TOKEN}"
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


def _folder_payload(
    *,
    folder_id: str = _FOLDER_ID,
    display_name: str = _DISPLAY_NAME,
    parent_folder_id: str | None = None,
    child_folder_count: int = 0,
    total_item_count: int = 0,
    unread_item_count: int = 0,
    is_hidden: bool = False,
    extra_field: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": folder_id,
        "displayName": display_name,
        "childFolderCount": child_folder_count,
        "totalItemCount": total_item_count,
        "unreadItemCount": unread_item_count,
        "isHidden": is_hidden,
    }
    if parent_folder_id is not None:
        payload["parentFolderId"] = parent_folder_id
    if extra_field is not None:
        payload["unknownField"] = extra_field
    return payload


def _reader(http: MagicMock) -> MsGraphMailFoldersReader:
    return MsGraphMailFoldersReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_folder(payload: dict[str, Any]) -> MsGraphMailFolder:
    return parse_msgraph_mail_folder(payload, expected_mailbox_user_id=_MAILBOX_USER_ID)


def _valid_folder(**overrides: object) -> MsGraphMailFolder:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _FOLDER_ID,
        "parent_remote_id": None,
        "display_name": _DISPLAY_NAME,
        "child_folder_count": 0,
        "total_item_count": 0,
        "unread_item_count": 0,
        "is_hidden": False,
    }
    defaults.update(overrides)
    return MsGraphMailFolder(**defaults)  # type: ignore[arg-type]


def _assert_safe_provider_error(exc: BaseException) -> None:
    assert str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc) == _SAFE_ERROR
    cause = exc.value.__cause__ if isinstance(exc, pytest.ExceptionInfo) else exc.__cause__
    assert cause is None
    message = str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc)
    for forbidden in (
        _MAILBOX_USER_ID,
        _FOLDER_ID,
        _DISPLAY_NAME,
        _HIDDEN_DISPLAY_NAME,
        "Authorization",
        "access token",
        "nextLink",
        _SECRET_TOKEN,
    ):
        assert forbidden not in message


# --- parser success ---


def test_parse_regular_root_folder() -> None:
    folder = _parse_folder(_folder_payload())
    assert folder.mailbox_user_id == _MAILBOX_USER_ID
    assert folder.remote_id == _FOLDER_ID
    assert folder.parent_remote_id is None
    assert folder.display_name == _DISPLAY_NAME
    assert folder.child_folder_count == 0
    assert folder.total_item_count == 0
    assert folder.unread_item_count == 0
    assert folder.is_hidden is False


def test_parse_nested_folder() -> None:
    folder = _parse_folder(
        _folder_payload(parent_folder_id=_PARENT_FOLDER_ID, child_folder_count=2)
    )
    assert folder.parent_remote_id == _PARENT_FOLDER_ID
    assert folder.child_folder_count == 2


def test_parse_hidden_folder() -> None:
    folder = _parse_folder(
        _folder_payload(display_name=_HIDDEN_DISPLAY_NAME, is_hidden=True)
    )
    assert folder.is_hidden is True
    assert folder.display_name == _HIDDEN_DISPLAY_NAME


def test_parse_zero_counts() -> None:
    folder = _parse_folder(_folder_payload())
    assert folder.total_item_count == 0
    assert folder.unread_item_count == 0


def test_parse_positive_counts() -> None:
    folder = _parse_folder(
        _folder_payload(total_item_count=10, unread_item_count=3, child_folder_count=1)
    )
    assert folder.total_item_count == 10
    assert folder.unread_item_count == 3
    assert folder.child_folder_count == 1


def test_parse_parent_folder_absent() -> None:
    folder = _parse_folder(_folder_payload())
    assert folder.parent_remote_id is None


def test_parse_display_name_trimmed() -> None:
    folder = _parse_folder(_folder_payload(display_name="  Inbox  "))
    assert folder.display_name == "Inbox"


def test_parse_expected_mailbox_user_id_preserved() -> None:
    folder = _parse_folder(_folder_payload())
    assert folder.mailbox_user_id == _MAILBOX_USER_ID


def test_parse_unknown_provider_fields_not_retained() -> None:
    folder = _parse_folder(_folder_payload(extra_field="should-not-appear"))
    assert not hasattr(folder, "unknownField")
    dumped = folder.model_dump()
    assert "unknownField" not in dumped


def test_display_name_hidden_from_repr() -> None:
    folder = _valid_folder(display_name=_DISPLAY_NAME)
    assert _DISPLAY_NAME not in repr(folder)


def test_raw_provider_payload_not_stored() -> None:
    payload = _folder_payload()
    folder = _parse_folder(payload)
    assert not hasattr(folder, "__pydantic_extra__") or not folder.__pydantic_extra__


# --- malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {},
        {"displayName": _DISPLAY_NAME},
        {"id": 123, "displayName": _DISPLAY_NAME},
        {"id": "", "displayName": _DISPLAY_NAME},
        {"id": _FOLDER_ID},
        {"id": _FOLDER_ID, "displayName": 123},
        {"id": _FOLDER_ID, "displayName": ""},
        {"id": _FOLDER_ID, "displayName": "bad\x00name"},
        {"id": _FOLDER_ID, "displayName": _DISPLAY_NAME, "childFolderCount": -1},
        {"id": _FOLDER_ID, "displayName": _DISPLAY_NAME, "childFolderCount": True},
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": -1,
        },
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": 0,
        },
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": 5,
            "unreadItemCount": 10,
        },
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": 0,
            "unreadItemCount": 0,
        },
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": 0,
            "unreadItemCount": 0,
            "isHidden": 1,
        },
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": 0,
            "unreadItemCount": 0,
            "isHidden": False,
            "parentFolderId": None,
        },
        {
            "id": _FOLDER_ID,
            "displayName": _DISPLAY_NAME,
            "childFolderCount": 0,
            "totalItemCount": 0,
            "unreadItemCount": 0,
            "isHidden": False,
            "parentFolderId": 123,
        },
    ],
)
def test_malformed_provider_payload_rejected(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_mail_folder(payload, expected_mailbox_user_id=_MAILBOX_USER_ID)
    _assert_safe_provider_error(exc)


# --- page model ---


def test_page_empty_tuple() -> None:
    page = MsGraphMailFolderPage(items=())
    assert page.items == ()
    assert page.has_more is False


def test_page_multiple_folders() -> None:
    page = MsGraphMailFolderPage(
        items=(
            _valid_folder(remote_id="f1"),
            _valid_folder(remote_id="f2", display_name="Sent"),
        )
    )
    assert len(page.items) == 2


def test_page_has_more_false() -> None:
    page = MsGraphMailFolderPage(items=())
    assert page.has_more is False


def test_page_has_more_true() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_root_next_link(),
    )
    page = MsGraphMailFolderPage(items=(), continuation=continuation)
    assert page.has_more is True


def test_page_duplicate_folder_ids_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphMailFolderPage(
            items=(
                _valid_folder(remote_id="same"),
                _valid_folder(remote_id="same", display_name="Other"),
            )
        )


def test_page_items_as_list_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphMailFolderPage(items=[_valid_folder()])  # type: ignore[arg-type]


def test_page_item_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphMailFolderPage(items=("not-a-folder",))  # type: ignore[arg-type]


def test_page_delta_continuation_rejected() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_root_next_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphMailFolderPage(items=(), continuation=delta)


def test_token_hidden_from_repr() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_root_next_link(),
    )
    page = MsGraphMailFolderPage(items=(), continuation=continuation)
    assert _SECRET_TOKEN not in repr(page)
    assert _SECRET_TOKEN not in repr(continuation)


@pytest.mark.parametrize(
    "folder_kwargs",
    [
        {"remote_id": None},
        {"child_folder_count": "x"},
        {"is_hidden": "yes"},
        {"unread_item_count": 5, "total_item_count": 1},
    ],
)
def test_malformed_folder_model_construct_rejected(folder_kwargs: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _FOLDER_ID,
        "parent_remote_id": None,
        "display_name": _DISPLAY_NAME,
        "child_folder_count": 0,
        "total_item_count": 0,
        "unread_item_count": 0,
        "is_hidden": False,
    }
    defaults.update(folder_kwargs)
    malformed = MsGraphMailFolder.model_construct(**defaults)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_folder(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_missing_items() -> None:
    malformed = MsGraphMailFolderPage.model_construct()
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_folder_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_items_as_list() -> None:
    malformed = MsGraphMailFolderPage.model_construct(items=[_valid_folder()])
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_folder_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_folder() -> None:
    bad_folder = MsGraphMailFolder.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_FOLDER_ID,
        display_name=_DISPLAY_NAME,
        child_folder_count=0,
        total_item_count=0,
        unread_item_count=0,
        is_hidden="no",  # type: ignore[arg-type]
    )
    malformed = MsGraphMailFolderPage.model_construct(items=(bad_folder,))
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_folder_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_continuation() -> None:
    bad_continuation = MsGraphKnowledgeContinuation.model_construct(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_root_next_link(),
    )
    malformed = MsGraphMailFolderPage.model_construct(
        items=(_valid_folder(),),
        continuation=bad_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_mail_folder_page(malformed)
    _assert_safe_provider_error(exc)


# --- request tests ---


def test_root_request_path_and_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    _reader(http).read_folders_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        parent_folder_id=None,
        continuation=None,
        limit=50,
    )
    call = http.get.call_args
    assert call.args[0] == _ROOT_PATH
    assert call.kwargs["params"]["$top"] == 50
    assert call.kwargs["params"]["$select"] == _SELECT
    assert call.kwargs["params"]["includeHiddenFolders"] == "true"
    assert "$expand" not in call.kwargs["params"]
    assert "messages" not in call.args[0]


def test_child_request_path_and_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    _reader(http).read_folders_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        parent_folder_id=_PARENT_FOLDER_ID,
        continuation=None,
        limit=25,
    )
    call = http.get.call_args
    assert call.args[0] == _CHILD_PATH
    assert call.kwargs["params"]["$top"] == 25
    assert call.kwargs["params"]["$select"] == _SELECT
    assert call.kwargs["params"]["includeHiddenFolders"] == "true"


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_root_next_link(),
    )
    _reader(http).read_folders_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        parent_folder_id=None,
        continuation=continuation,
        limit=100,
    )
    assert http.get.call_args.args[0] == _root_next_link()
    assert "params" not in http.get.call_args.kwargs


@pytest.mark.parametrize("limit", [0, 201, True, "50"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_folders_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            parent_folder_id=None,
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


@pytest.mark.parametrize("mailbox_user_id", ["", "  ", "bad\x00id", 123])
def test_invalid_mailbox_user_id_rejected_before_http(mailbox_user_id: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_folders_page(
            mailbox_user_id=mailbox_user_id,  # type: ignore[arg-type]
            parent_folder_id=None,
            continuation=None,
            limit=100,
        )
    http.get.assert_not_called()


@pytest.mark.parametrize("parent_folder_id", ["", "bad\x00id", 123])
def test_invalid_parent_folder_id_rejected_before_http(parent_folder_id: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_folders_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            parent_folder_id=parent_folder_id,  # type: ignore[arg-type]
            continuation=None,
            limit=100,
        )
    http.get.assert_not_called()


# --- continuation tests ---


def test_validate_root_continuation_same_user() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_root_next_link(),
    )
    validated = validate_msgraph_mail_folders_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        parent_folder_id=None,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated is continuation


def test_validate_child_continuation_same_user_and_parent() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_child_next_link(),
    )
    validated = validate_msgraph_mail_folders_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        parent_folder_id=_PARENT_FOLDER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated is continuation


@pytest.mark.parametrize(
    ("url", "parent_folder_id"),
    [
        (
            f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX_USER_ID, safe='')}/"
            f"mailFolders?$skiptoken={_SECRET_TOKEN}",
            None,
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
            f"{quote(_OTHER_FOLDER_ID, safe='')}/childFolders?$skiptoken={_SECRET_TOKEN}",
            _PARENT_FOLDER_ID,
        ),
        (_root_next_link(), _PARENT_FOLDER_ID),
        (_child_next_link(), None),
        (f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages?$skiptoken=x", None),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/delta?"
            f"$deltatoken=x",
            None,
        ),
        ("https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x", None),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendar/events?"
            f"$skiptoken=x",
            None,
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
            f"{_QUOTED_FOLDER_ID}/childFolders/extra?$skiptoken={_SECRET_TOKEN}",
            _FOLDER_ID,
        ),
    ],
)
def test_rejects_invalid_mail_folders_continuation(
    url: str,
    parent_folder_id: str | None,
) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_folders_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            parent_folder_id=parent_folder_id,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert exc.value.__cause__ is None


def test_delta_continuation_rejected_in_validator() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_root_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_mail_folders_continuation(
            delta,
            mailbox_user_id=_MAILBOX_USER_ID,
            parent_folder_id=None,
            graph_base_url=_GRAPH_BASE,
        )


def test_invalid_continuation_rejected_before_http() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/"
            f"{quote(_OTHER_MAILBOX_USER_ID, safe='')}/mailFolders?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        _reader(http).read_folders_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            parent_folder_id=None,
            continuation=continuation,
            limit=100,
        )
    http.get.assert_not_called()


# --- delegation ---


def test_graph_rest_client_delegates_mail_folders() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(value=[_folder_payload()])
    )
    page = _graph_client(http).read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert len(page.items) == 1
    assert page.items[0].remote_id == _FOLDER_ID


def test_collaboration_suite_delegates_mail_folders() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_integration_delegates_mail_folders() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert client._knowledge_transport._http_client is http
    assert client._mail_folders_reader._transport._http_client is http
    http.get.assert_called_once()


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert client._http_client is http


def test_existing_drive_operations_still_work() -> None:
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


def test_existing_list_messages_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload={"value": []})
    client = _graph_client(http)
    result = client.list_messages(_MAILBOX_USER_ID, folder="inbox", limit=5)
    assert result.messages == []


def test_existing_get_message_still_works() -> None:
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


class _CustomSuiteWithoutMailFolders(CollaborationSuite):
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


def test_custom_client_without_mail_folders_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutMailFolders(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph integration does not expose Mail folders knowledge capability",
    ):
        integration.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)


class _CustomMailFoldersSuite(CollaborationSuite):
    def __init__(self, page: MsGraphMailFolderPage) -> None:
        self._page = page

    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None = None,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphMailFolderPage:
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


def test_custom_client_malformed_page_rejected() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomMailFoldersSuite(page=MsGraphMailFolderPage.model_construct()),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = MsGraphMailFolderPage(items=(_valid_folder(),))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomMailFoldersSuite(page=supplied),
        enabled=True,
    )
    returned = integration.read_mail_folders_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


# --- security ---


def test_security_folder_repr_and_errors() -> None:
    folder = _valid_folder(display_name=_DISPLAY_NAME)
    assert _DISPLAY_NAME not in repr(folder)
    assert _FOLDER_ID in repr(folder)

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_folders_page(
            mailbox_user_id="",
            parent_folder_id=None,
            continuation=None,
            limit=100,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)

    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_mail_folders_continuation(
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_root_next_link(),
            ),
            mailbox_user_id=_MAILBOX_USER_ID,
            parent_folder_id=_PARENT_FOLDER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
