# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Drive knowledge-read delta surface."""

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
    MsGraphDriveDeltaPage,
    MsGraphDriveItemKind,
    MsGraphDriveKnowledgeReader,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_drive_item,
    validate_msgraph_drive_delta_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_DRIVE_ID = "b!drive-id-with-special-chars"
_QUOTED_DRIVE_ID = quote(_DRIVE_ID, safe="")
_SECRET_TOKEN = "super-secret-delta-token-value"
_NEXT_LINK = (
    f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
    f"$skiptoken={_SECRET_TOKEN}"
)
_DELTA_LINK = (
    f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
    f"$deltatoken={_SECRET_TOKEN}"
)
_OTHER_DRIVE_NEXT = (
    "https://graph.microsoft.com/v1.0/drives/other-drive/root/delta?"
    "$skiptoken=other-token"
)
_MAIL_NEXT = (
    "https://graph.microsoft.com/v1.0/users/user-1/messages?"
    "$skiptoken=mail-token"
)
_TS = "2026-05-29T10:15:30Z"
_SELECT = (
    "id,name,parentReference,webUrl,eTag,cTag,size,file,folder,package,deleted,root,"
    "createdDateTime,lastModifiedDateTime"
)


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _active_item_payload(
    *,
    item_id: str = "item-1",
    name: str = "report.pdf",
    kind: str = "file",
    parent_id: str | None = "parent-1",
    mime_type: str = "application/pdf",
    size: int | None = 1024,
    include_root: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": item_id,
        "name": name,
        "eTag": '"etag-1"',
        "cTag": '"ctag-1"',
        "size": size,
        "webUrl": "https://contoso.sharepoint.com/file",
        "createdDateTime": _TS,
        "lastModifiedDateTime": _TS,
    }
    if parent_id is not None:
        payload["parentReference"] = {"id": parent_id, "driveId": _DRIVE_ID}
    if kind == "file":
        payload["file"] = {"mimeType": mime_type, "hashes": {"quickXorHash": "abc"}}
    elif kind == "folder":
        payload["folder"] = {"childCount": 1}
    elif kind == "package":
        payload["package"] = {"type": "oneNote"}
    if include_root:
        payload["root"] = {}
    return payload


def _page_payload(
    *,
    items: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
    delta_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": items or []}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    if delta_link is not None:
        payload["@odata.deltaLink"] = delta_link
    return payload


def _mock_http(*, status_code: int = 200, json_payload: object | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_payload if json_payload is not None else {}
    response.raise_for_status = MagicMock()
    client.get.return_value = response
    return client


def _reader(http: MagicMock | None = None) -> MsGraphDriveKnowledgeReader:
    http_client = http or _mock_http()
    transport = MsGraphKnowledgeTransport(_config(), http_client=http_client)
    return MsGraphDriveKnowledgeReader(_config(), transport=transport)


# --- drive item parser ---


def test_parse_file_item() -> None:
    item = parse_msgraph_drive_item(
        _active_item_payload(item_id="file-1", name="notes.txt", mime_type="text/plain"),
        expected_drive_id=_DRIVE_ID,
    )
    assert item.remote_id == "file-1"
    assert item.drive_id == _DRIVE_ID
    assert item.kind == MsGraphDriveItemKind.FILE
    assert item.name == "notes.txt"
    assert item.mime_type == "text/plain"
    assert item.size_bytes == 1024
    assert item.last_modified_at == datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc)


def test_parse_folder_item() -> None:
    item = parse_msgraph_drive_item(
        _active_item_payload(item_id="folder-1", name="Docs", kind="folder"),
        expected_drive_id=_DRIVE_ID,
    )
    assert item.kind == MsGraphDriveItemKind.FOLDER
    assert item.mime_type is None


def test_parse_package_item() -> None:
    item = parse_msgraph_drive_item(
        _active_item_payload(item_id="pkg-1", name="Notebook", kind="package"),
        expected_drive_id=_DRIVE_ID,
    )
    assert item.kind == MsGraphDriveItemKind.PACKAGE


def test_parse_other_item_without_facets() -> None:
    payload = _active_item_payload(item_id="other-1", name="Unknown", kind="file")
    del payload["file"]
    item = parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)
    assert item.kind == MsGraphDriveItemKind.OTHER


def test_parse_root_folder_without_parent() -> None:
    payload = _active_item_payload(
        item_id="root-1",
        name="root",
        kind="folder",
        parent_id=None,
        include_root=True,
    )
    item = parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)
    assert item.is_root is True
    assert item.parent_remote_id is None


def test_parse_minimal_deleted_item() -> None:
    item = parse_msgraph_drive_item(
        {"id": "deleted-1", "deleted": {}},
        expected_drive_id=_DRIVE_ID,
    )
    assert item.kind == MsGraphDriveItemKind.DELETED
    assert item.remote_id == "deleted-1"
    assert item.name is None
    assert item.e_tag is None
    assert item.last_modified_at is None


def test_parse_deleted_without_name_tags_or_timestamps() -> None:
    item = parse_msgraph_drive_item(
        {"id": "deleted-2", "deleted": {"state": "deleted"}},
        expected_drive_id=_DRIVE_ID,
    )
    assert item.kind == MsGraphDriveItemKind.DELETED
    assert item.deleted_state == "deleted"
    assert item.name is None
    assert item.c_tag is None


def test_parse_rejects_cross_drive_parent_reference() -> None:
    payload = _active_item_payload()
    payload["parentReference"]["driveId"] = "other-drive"
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response") as exc:
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)
    assert _DRIVE_ID not in str(exc.value)
    assert "other-drive" not in str(exc.value)


def test_parse_rejects_missing_id() -> None:
    payload = _active_item_payload()
    del payload["id"]
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_non_string_id() -> None:
    payload = _active_item_payload()
    payload["id"] = 123
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_empty_id() -> None:
    payload = _active_item_payload()
    payload["id"] = "   "
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_file_and_folder_facets() -> None:
    payload = _active_item_payload(kind="file")
    payload["folder"] = {"childCount": 0}
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_negative_size() -> None:
    payload = _active_item_payload(size=-1)
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_bool_size() -> None:
    payload = _active_item_payload()
    payload["size"] = True
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_malformed_timestamp() -> None:
    payload = _active_item_payload()
    payload["lastModifiedDateTime"] = "not-a-date"
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_naive_timestamp() -> None:
    payload = _active_item_payload()
    payload["lastModifiedDateTime"] = "2026-05-29T10:15:30"
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_rejects_non_string_mime_type() -> None:
    payload = _active_item_payload()
    payload["file"] = {"mimeType": 42}
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)


def test_parse_does_not_retain_raw_provider_payload() -> None:
    payload = _active_item_payload()
    item = parse_msgraph_drive_item(payload, expected_drive_id=_DRIVE_ID)
    assert not hasattr(item, "file")
    assert not hasattr(item, "folder")
    assert not hasattr(item, "deleted")
    model_dump = item.model_dump()
    assert "hashes" not in str(model_dump)
    assert "quickXorHash" not in str(model_dump)


def test_parse_error_does_not_leak_id_or_payload() -> None:
    secret_id = "secret-item-id-value"
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response") as exc:
        parse_msgraph_drive_item({"id": secret_id, "name": ""}, expected_drive_id=_DRIVE_ID)
    assert secret_id not in str(exc.value)
    assert exc.value.__cause__ is None


def test_drive_item_repr_hides_sensitive_fields() -> None:
    item = parse_msgraph_drive_item(_active_item_payload(), expected_drive_id=_DRIVE_ID)
    rendered = repr(item)
    assert '"etag-1"' not in rendered
    assert '"ctag-1"' not in rendered
    assert "sharepoint.com" not in rendered


# --- delta page ---


def test_delta_page_first_page_with_next_link() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[_active_item_payload(item_id="a-1", name="A")],
            next_link=_NEXT_LINK,
        )
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    assert page.has_more is True
    assert page.is_complete is False
    assert page.continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert len(page.items) == 1


def test_delta_page_last_page_with_delta_link() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[_active_item_payload(item_id="b-1", name="B")],
            delta_link=_DELTA_LINK,
        )
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    assert page.has_more is False
    assert page.is_complete is True
    assert page.continuation.kind == MsGraphKnowledgeContinuationKind.DELTA


def test_delta_page_multiple_files_and_folders() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[
                _active_item_payload(item_id="f-1", name="File", kind="file"),
                _active_item_payload(item_id="d-1", name="Folder", kind="folder"),
            ],
            delta_link=_DELTA_LINK,
        )
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    kinds = {item.kind for item in page.items}
    assert kinds == {MsGraphDriveItemKind.FILE, MsGraphDriveItemKind.FOLDER}


def test_delta_page_includes_deleted_tombstone() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[{"id": "gone-1", "deleted": {"state": "deleted"}}],
            delta_link=_DELTA_LINK,
        )
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    assert page.items[0].kind == MsGraphDriveItemKind.DELETED


def test_delta_page_duplicate_id_last_occurrence_wins() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[
                _active_item_payload(item_id="dup-1", name="Version 1"),
                _active_item_payload(item_id="keep-1", name="Keep"),
                {"id": "dup-1", "deleted": {"state": "deleted"}},
            ],
            delta_link=_DELTA_LINK,
        )
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    assert [item.remote_id for item in page.items] == ["keep-1", "dup-1"]
    assert page.items[1].kind == MsGraphDriveItemKind.DELETED


def test_delta_page_malformed_common_page() -> None:
    http = _mock_http(json_payload={"value": "not-a-list"})
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)


def test_delta_page_requires_continuation_in_delta_mode() -> None:
    http = _mock_http(json_payload=_page_payload(items=[_active_item_payload()]))
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)


def test_delta_page_hides_token_in_repr() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[_active_item_payload()],
            next_link=_NEXT_LINK,
        )
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    rendered = repr(page)
    assert _SECRET_TOKEN not in rendered
    assert "nextLink" not in rendered
    assert "skiptoken" not in rendered


def test_delta_page_error_does_not_leak_token() -> None:
    http = _mock_http(json_payload={"value": "not-a-list"})
    with pytest.raises(ValueError) as exc:
        _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=50)
    assert _SECRET_TOKEN not in str(exc.value)


# --- requests ---


def test_first_page_uses_quoted_drive_delta_path() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=25)
    expected_path = f"/drives/{_QUOTED_DRIVE_ID}/root/delta"
    assert http.get.call_args.args[0] == expected_path


def test_first_page_passes_top_via_params() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=25)
    assert http.get.call_args.kwargs["params"]["$top"] == 25


def test_first_page_uses_explicit_select() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=25)
    assert http.get.call_args.kwargs["params"]["$select"] == _SELECT


def test_continuation_uses_full_url_without_params() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_NEXT_LINK,
    )
    _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=continuation, limit=25)
    assert http.get.call_args.args[0] == _NEXT_LINK
    assert "params" not in http.get.call_args.kwargs or http.get.call_args.kwargs.get("params") is None


def test_delta_continuation_starts_next_round() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_DELTA_LINK,
    )
    page = _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=continuation, limit=25)
    assert page.is_complete is True
    assert http.get.call_args.args[0] == _DELTA_LINK


def test_rejects_other_drive_continuation_before_http() -> None:
    http = _mock_http()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_OTHER_DRIVE_NEXT,
    )
    with pytest.raises(IntegrationConfigurationError, match="invalid Microsoft Graph drive continuation") as exc:
        _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=continuation, limit=25)
    assert _SECRET_TOKEN not in str(exc.value)
    assert _DRIVE_ID not in str(exc.value)
    http.get.assert_not_called()


def test_rejects_mail_continuation_before_http() -> None:
    http = _mock_http()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_MAIL_NEXT,
    )
    with pytest.raises(IntegrationConfigurationError, match="invalid Microsoft Graph drive continuation"):
        _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=continuation, limit=25)
    http.get.assert_not_called()


@pytest.mark.parametrize("limit", [0, 201, True, "25"])
def test_rejects_invalid_limit(limit: object) -> None:
    http = _mock_http()
    with pytest.raises(IntegrationConfigurationError, match="invalid Microsoft Graph drive delta page limit"):
        _reader(http).read_delta_page(drive_id=_DRIVE_ID, continuation=None, limit=limit)  # type: ignore[arg-type]
    http.get.assert_not_called()


def test_transport_and_reader_share_injected_http_client() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    client = GraphRestClient(_config(), http_client=http)
    client.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)
    assert client._knowledge_transport._http_client is http
    assert client._drive_knowledge_reader._transport._http_client is http
    http.get.assert_called_once()


# --- continuation validation ---


def test_validate_drive_delta_continuation_accepts_next_page() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_NEXT_LINK,
    )
    validated = validate_msgraph_drive_delta_continuation(
        continuation,
        drive_id=_DRIVE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated is continuation


def test_validate_drive_delta_continuation_accepts_delta() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_DELTA_LINK,
    )
    validated = validate_msgraph_drive_delta_continuation(
        continuation,
        drive_id=_DRIVE_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated.kind == MsGraphKnowledgeContinuationKind.DELTA


# --- integration delegation ---


def test_graph_rest_client_delegates_drive_delta() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[_active_item_payload(item_id="x-1", name="X")],
            delta_link=_DELTA_LINK,
        )
    )
    client = GraphRestClient(_config(), http_client=http)
    page = client.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)
    assert isinstance(page, MsGraphDriveDeltaPage)
    assert page.items[0].remote_id == "x-1"


def test_collaboration_suite_delegates_drive_delta() -> None:
    http = _mock_http(json_payload=_page_payload(delta_link=_DELTA_LINK))
    client = GraphRestClient(_config(), http_client=http)
    suite = _Ms365GraphCollaborationSuite(client)
    suite.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)
    assert http.get.call_args.args[0] == f"/drives/{_QUOTED_DRIVE_ID}/root/delta"


def test_integration_delegates_drive_delta() -> None:
    http = _mock_http(
        json_payload=_page_payload(
            items=[_active_item_payload(item_id="y-1", name="Y")],
            delta_link=_DELTA_LINK,
        )
    )
    client = GraphRestClient(_config(), http_client=http)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    page = integration.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)
    assert page.items[0].name == "Y"


class _MailOnlyCollaborationSuite(CollaborationSuite):
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


def test_custom_collaboration_suite_without_drive_still_constructible() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _MailOnlyCollaborationSuite(),
        enabled=True,
    )
    assert integration.client is not None


def test_custom_collaboration_suite_drive_call_fails_safely() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _MailOnlyCollaborationSuite(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="does not expose Drive knowledge capability",
    ):
        integration.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)


def test_existing_mail_operation_unchanged_on_graph_integration() -> None:
    http = MagicMock()
    response = MagicMock()
    response.json.return_value = {
        "id": "msg-1",
        "subject": "Hello",
        "bodyPreview": "Preview",
        "from": {"emailAddress": {"address": "sender@example.com"}},
        "receivedDateTime": "2026-05-29T10:00:00Z",
    }
    response.raise_for_status.return_value = None
    http.get.return_value = response
    client = GraphRestClient(_config(), http_client=http)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    message = integration.get_message("user@example.com", "msg-1")
    assert message.subject == "Hello"
