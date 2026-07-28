# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Drive knowledge-read content download surface."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import MagicMock, patch
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
    ABSOLUTE_DRIVE_CONTENT_MAX_BYTES,
    DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    MsGraphDriveContentChanged,
    MsGraphDriveContentReader,
    MsGraphDriveContentTooLarge,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    MsGraphKnowledgeTransport,
    validate_msgraph_drive_download_url,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_DRIVE_ID = "b!drive-id-with-special-chars"
_ITEM_ID = "item-abc-123"
_QUOTED_DRIVE_ID = quote(_DRIVE_ID, safe="")
_QUOTED_ITEM_ID = quote(_ITEM_ID, safe="")
_CTAG = '"ctag-content-1"'
_TS = "2026-05-29T10:15:30Z"
_DOWNLOAD_URL = (
    "https://contoso-my.sharepoint.com/personal/user/_layouts/15/download.aspx?"
    "UniqueId=abc&translate=false"
)
_SECRET_LOCATION = f"https://secret-cdn.example.com/download?token={_DOWNLOAD_URL.split('?')[1]}"
_SELECT = (
    "id,name,parentReference,webUrl,eTag,cTag,size,file,createdDateTime,lastModifiedDateTime"
)
_CONTENT_PATH = f"/drives/{_QUOTED_DRIVE_ID}/items/{_QUOTED_ITEM_ID}/content"
_METADATA_PATH = f"/drives/{_QUOTED_DRIVE_ID}/items/{_QUOTED_ITEM_ID}"


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _file_payload(
    *,
    item_id: str = _ITEM_ID,
    c_tag: str = _CTAG,
    size: int | None = 42,
    mime_type: str = "application/pdf",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "name": "report.pdf",
        "eTag": '"etag-1"',
        "cTag": c_tag,
        "size": size,
        "webUrl": "https://contoso.sharepoint.com/file",
        "createdDateTime": _TS,
        "lastModifiedDateTime": _TS,
        "parentReference": {"id": "parent-1", "driveId": _DRIVE_ID},
        "file": {"mimeType": mime_type},
    }


def _drive_item(
    *,
    c_tag: str = _CTAG,
    size: int | None = 11,
    mime_type: str = "application/pdf",
) -> MsGraphDriveItem:
    return MsGraphDriveItem(
        remote_id=_ITEM_ID,
        drive_id=_DRIVE_ID,
        parent_remote_id="parent-1",
        kind=MsGraphDriveItemKind.FILE,
        name="report.pdf",
        e_tag='"etag-1"',
        c_tag=c_tag,
        size_bytes=size,
        mime_type=mime_type,
        created_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
        last_modified_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
        web_url="https://contoso.sharepoint.com/file",
    )


def _json_response(*, status_code: int = 200, payload: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status = MagicMock()
    return response


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


def _content_reader(
    graph_http: MagicMock,
    download_http: MagicMock,
) -> MsGraphDriveContentReader:
    transport = MsGraphKnowledgeTransport(_config(), http_client=graph_http)
    return MsGraphDriveContentReader(
        _config(),
        graph_transport=transport,
        graph_http_client=graph_http,
        download_http_client=download_http,
    )


def _graph_client(
    graph_http: MagicMock,
    download_http: MagicMock | None = None,
) -> GraphRestClient:
    return GraphRestClient(
        _config(),
        http_client=graph_http,
        download_http_client=download_http,
    )


def _setup_happy_path(
    *,
    file_bytes: bytes = b"hello-world",
    c_tag: str = _CTAG,
    size: int | None = None,
) -> tuple[MagicMock, MagicMock, MsGraphDriveContentReader]:
    if size is None:
        size = len(file_bytes)
    graph_http = MagicMock()
    metadata_payload = _file_payload(c_tag=c_tag, size=size)
    redirect_response = MagicMock()
    redirect_response.status_code = 302
    redirect_response.headers = {"Location": _DOWNLOAD_URL}

    graph_http.get.side_effect = [
        _json_response(payload=metadata_payload),
        redirect_response,
        _json_response(payload=metadata_payload),
    ]

    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(file_bytes))},
        chunks=(file_bytes,),
    )
    reader = _content_reader(graph_http, download_http)
    return graph_http, download_http, reader


# --- download URL validation ---


@pytest.mark.parametrize(
    "url",
    [
        _DOWNLOAD_URL,
        "https://files.example.com/path/to/file.bin?sig=abc%3D%3D&other=1",
    ],
)
def test_validate_download_url_accepts_valid_https(url: str) -> None:
    validated = validate_msgraph_drive_download_url(url)
    assert validated == url


def test_validate_download_url_preserves_query_string() -> None:
    url = "https://cdn.example.com/dl/file?a=1&b=two%20three"
    assert validate_msgraph_drive_download_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "/relative/path",
        "http://example.com/file",
        "https://user:pass@example.com/file",
        "https://example.com/file#frag",
        "https://localhost/file",
        "https://host.local/file",
        "https://host.internal/file",
        "https://127.0.0.1/file",
        "https://[::1]/file",
        "https://example.com:8443/file",
        "https://example.com/\x00evil",
        "https://example.com",
        "https://example.com:invalid/file",
        "https://example.com:99999/file",
        "https://[broken-ipv6/file",
        "https://localhost./file",
        "https://files.localhost/file",
        "https://127.1/file",
        "https://2130706433/file",
        "https://singlelabel/file",
    ],
)
def test_validate_download_url_rejects_invalid(url: str) -> None:
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download location is invalid",
    ) as exc:
        validate_msgraph_drive_download_url(url)
    assert url not in str(exc.value)
    assert exc.value.__cause__ is None


# --- custom item boundary (model_construct) ---


def _construct_file_item(**overrides: object) -> MsGraphDriveItem:
    base: dict[str, object] = {
        "remote_id": _ITEM_ID,
        "drive_id": _DRIVE_ID,
        "parent_remote_id": "parent-1",
        "kind": MsGraphDriveItemKind.FILE,
        "name": "report.pdf",
        "e_tag": '"etag-1"',
        "c_tag": _CTAG,
        "size_bytes": 11,
        "mime_type": "application/pdf",
        "created_at": datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
        "last_modified_at": datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
        "web_url": "https://contoso.sharepoint.com/file",
    }
    base.update(overrides)
    return MsGraphDriveItem.model_construct(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"c_tag": 123},
        {"size_bytes": "100"},
        {"size_bytes": True},
        {"size_bytes": -1},
        {"kind": "file"},
        {"remote_id": 123},
        {"drive_id": ""},
        {"mime_type": 123},
    ],
)
def test_model_construct_malformed_item_field_rejected(overrides: dict[str, object]) -> None:
    graph_http = MagicMock()
    download_http = MagicMock()
    reader = _content_reader(graph_http, download_http)
    item = _construct_file_item(**overrides)
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response") as exc:
        reader.read_file_content(item=item, max_bytes=1024)
    assert exc.value.__cause__ is None
    graph_http.get.assert_not_called()
    download_http.stream.assert_not_called()


def test_model_construct_missing_kind_rejected() -> None:
    graph_http = MagicMock()
    download_http = MagicMock()
    reader = _content_reader(graph_http, download_http)
    item = MsGraphDriveItem.model_construct(
        remote_id=_ITEM_ID,
        drive_id=_DRIVE_ID,
        c_tag=_CTAG,
        last_modified_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response") as exc:
        reader.read_file_content(item=item, max_bytes=1024)
    assert exc.value.__cause__ is None
    graph_http.get.assert_not_called()
    download_http.stream.assert_not_called()


def test_model_construct_missing_remote_id_rejected() -> None:
    graph_http = MagicMock()
    download_http = MagicMock()
    reader = _content_reader(graph_http, download_http)
    item = MsGraphDriveItem.model_construct(
        drive_id=_DRIVE_ID,
        kind=MsGraphDriveItemKind.FILE,
        c_tag=_CTAG,
        last_modified_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response") as exc:
        reader.read_file_content(item=item, max_bytes=1024)
    assert exc.value.__cause__ is None
    graph_http.get.assert_not_called()
    download_http.stream.assert_not_called()


# --- malformed redirect headers ---


class _BrokenHeaderMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Iterator[tuple[str, str]]:
        raise RuntimeError("broken headers")


def test_content_redirect_integer_header_key_rejected() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=302, headers={1: "x"}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ) as exc:
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert exc.value.__cause__ is None
    assert "Location" not in str(exc.value)


def test_content_redirect_bytes_header_key_rejected() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=302, headers={b"Location": _DOWNLOAD_URL}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ) as exc:
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert exc.value.__cause__ is None


def test_content_redirect_duplicate_location_different_case_rejected() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(
            status_code=302,
            headers={"Location": _DOWNLOAD_URL, "location": _SECRET_LOCATION},
        ),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ) as exc:
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert exc.value.__cause__ is None
    assert _SECRET_LOCATION not in str(exc.value)


def test_content_redirect_headers_items_raises_rejected() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=302, headers=_BrokenHeaderMapping()),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ) as exc:
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert exc.value.__cause__ is None


# --- malformed download headers ---


def test_download_content_length_integer_header_key_rejected() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.return_value = _FakeStreamContext(
        headers={1: "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download response is invalid",
    ) as exc:
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_duplicate_content_length_different_case_rejected() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5", "content-length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download response is invalid",
    ) as exc:
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_headers_items_raises_rejected() -> None:
    graph_http, download_http, _ = _setup_happy_path(file_bytes=b"hello")
    download_http.stream.return_value = _FakeStreamContext(
        headers=_BrokenHeaderMapping(),
        chunks=(b"hello",),
    )
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download response is invalid",
    ) as exc:
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(size=5),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


# --- HTTP clients (opens) ---


def test_graph_client_has_auth_base_url_no_follow_redirects() -> None:
    with patch("httpx.Client") as client_cls:
        mock_client = MagicMock()
        client_cls.return_value = mock_client
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.opens import (
            open_graph_rest_client,
        )

        open_graph_rest_client(_config(), access_token="test-token")

    graph_call = client_cls.call_args_list[0]
    kwargs = graph_call.kwargs
    assert kwargs["follow_redirects"] is False
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["base_url"] == _GRAPH_BASE.rstrip("/")


def test_download_client_has_no_auth_no_base_url() -> None:
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.opens import (
        _create_download_http_client,
    )

    with patch("httpx.Client") as client_cls:
        mock_client = MagicMock()
        client_cls.return_value = mock_client
        _create_download_http_client(_config())

    kwargs = client_cls.call_args.kwargs
    assert kwargs["follow_redirects"] is False
    assert kwargs["headers"]["Accept"] == "application/octet-stream"
    assert "base_url" not in kwargs
    assert "Authorization" not in kwargs.get("headers", {})


def test_opens_does_not_fetch_second_token_when_access_token_given() -> None:
    with patch("httpx.Client") as client_cls, patch(
        "intergrax.integrations.providers.collaboration_suite.ms365_graph.opens._fetch_access_token",
    ) as fetch_token:
        client_cls.return_value = MagicMock()
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.opens import (
            open_graph_rest_client,
        )

        open_graph_rest_client(_config(), access_token="injected-token")

    fetch_token.assert_not_called()


def test_injected_graph_without_download_client_delta_still_works() -> None:
    graph_http = MagicMock()
    graph_http.get.return_value = _json_response(
        payload={
            "value": [],
            "@odata.deltaLink": (
                f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
                "$deltatoken=tok"
            ),
        }
    )
    client = GraphRestClient(_config(), http_client=graph_http)
    page = client.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)
    assert page.is_complete is True


def test_injected_graph_without_download_client_content_fails_before_http() -> None:
    client = GraphRestClient(_config(), http_client=MagicMock())
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph Drive download client is not configured",
    ):
        client.read_drive_file_content(item=_drive_item())


def test_both_clients_injected_content_uses_instances() -> None:
    graph_http, download_http, reader = _setup_happy_path()
    client = GraphRestClient(
        _config(),
        http_client=graph_http,
        download_http_client=download_http,
    )
    result = client.read_drive_file_content(item=_drive_item())
    assert result.data == b"hello-world"
    assert graph_http.get.called
    assert download_http.stream.called


# --- redirect ---


def test_content_redirect_exact_path_and_headers() -> None:
    graph_http, download_http, reader = _setup_happy_path()
    reader.read_file_content(item=_drive_item(), max_bytes=1024)

    redirect_call = graph_http.get.call_args_list[1]
    assert redirect_call.args[0] == _CONTENT_PATH
    assert redirect_call.kwargs["follow_redirects"] is False
    assert redirect_call.kwargs["headers"] == {"Accept": "application/octet-stream"}
    assert "params" not in redirect_call.kwargs


def test_content_redirect_uses_quoted_ids() -> None:
    graph_http, _, reader = _setup_happy_path()
    reader.read_file_content(item=_drive_item(), max_bytes=1024)
    redirect_path = graph_http.get.call_args_list[1].args[0]
    assert _QUOTED_DRIVE_ID in redirect_path
    assert _QUOTED_ITEM_ID in redirect_path


@pytest.mark.parametrize("status_code", [200, 301, 307, 308])
def test_content_redirect_rejects_non_302(status_code: int) -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=status_code, headers={}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)


def test_content_redirect_missing_location() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=302, headers={}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ) as exc:
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert _SECRET_LOCATION not in str(exc.value)


def test_content_redirect_location_wrong_type() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=302, headers={"Location": 123}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive content redirect response is invalid",
    ):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)


@pytest.mark.parametrize("status_code", [401, 403])
def test_content_redirect_configuration_errors(status_code: int) -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=status_code, headers={}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(IntegrationConfigurationError):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)


@pytest.mark.parametrize("status_code", [404, 429, 500, 503])
def test_content_redirect_dependency_errors(status_code: int) -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        MagicMock(status_code=status_code, headers={}),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(IntegrationDependencyError):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)


def test_content_redirect_transport_exception_no_cause() -> None:
    graph_http = MagicMock()
    graph_http.get.side_effect = [
        _json_response(payload=_file_payload()),
        RuntimeError("network down"),
    ]
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(IntegrationDependencyError) as exc:
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert exc.value.__cause__ is None


def test_authenticated_client_does_not_download_bytes_directly() -> None:
    data = b"from-cdn"
    graph_http, download_http, reader = _setup_happy_path(file_bytes=data)
    reader.read_file_content(item=_drive_item(size=len(data)), max_bytes=1024)
    redirect_call = graph_http.get.call_args_list[1]
    assert redirect_call.args[0].endswith("/content")
    download_http.stream.assert_called_once()


# --- streaming ---


def test_download_small_file_with_sha256_and_mime() -> None:
    data = b"small-file-content"
    _, _, reader = _setup_happy_path(file_bytes=data)
    result = reader.read_file_content(
        item=_drive_item(size=len(data), mime_type="text/plain"),
        max_bytes=1024,
    )
    assert isinstance(result, MsGraphDriveFileContent)
    assert result.data == data
    assert result.size_bytes == len(data)
    assert result.mime_type == "text/plain"
    assert result.content_hash == hashlib.sha256(data).hexdigest()
    assert result.content_revision == _CTAG


def test_download_empty_file() -> None:
    _, _, reader = _setup_happy_path(file_bytes=b"", size=0)
    result = reader.read_file_content(item=_drive_item(size=0), max_bytes=1024)
    assert result.data == b""
    assert result.size_bytes == 0


def test_download_multiple_chunks() -> None:
    graph_http = MagicMock()
    metadata_payload = _file_payload(size=11)
    redirect_response = MagicMock(status_code=302, headers={"Location": _DOWNLOAD_URL})
    graph_http.get.side_effect = [
        _json_response(payload=metadata_payload),
        redirect_response,
        _json_response(payload=metadata_payload),
    ]
    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "11"},
        chunks=(b"hello", b"-world"),
    )
    result = _content_reader(graph_http, download_http).read_file_content(
        item=_drive_item(size=11),
        max_bytes=1024,
    )
    assert result.data == b"hello-world"


def test_download_without_content_length() -> None:
    graph_http = MagicMock()
    metadata_payload = _file_payload(size=5)
    redirect_response = MagicMock(status_code=302, headers={"Location": _DOWNLOAD_URL})
    graph_http.get.side_effect = [
        _json_response(payload=metadata_payload),
        redirect_response,
        _json_response(payload=metadata_payload),
    ]
    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(chunks=(b"12345",))
    result = _content_reader(graph_http, download_http).read_file_content(
        item=_drive_item(size=5),
        max_bytes=1024,
    )
    assert result.data == b"12345"


def test_download_content_length_exceeds_limit() -> None:
    graph_http = MagicMock()
    metadata_payload = _file_payload(size=5)
    redirect_response = MagicMock(status_code=302, headers={"Location": _DOWNLOAD_URL})
    graph_http.get.side_effect = [
        _json_response(payload=metadata_payload),
        redirect_response,
    ]
    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "100"},
        chunks=(b"x",),
    )
    with pytest.raises(MsGraphDriveContentTooLarge):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(size=5),
            max_bytes=10,
        )


def test_download_malformed_content_length() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "not-a-number"},
        chunks=(b"x",),
    )
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download response is invalid",
    ):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )


def test_download_bytes_exceed_limit_during_stream() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.return_value = _FakeStreamContext(
        chunks=(b"a" * 5, b"b" * 10),
    )
    with pytest.raises(MsGraphDriveContentTooLarge):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(size=15),
            max_bytes=10,
        )


def test_download_content_length_mismatch() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "10"},
        chunks=(b"short",),
    )
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download response is invalid",
    ):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )


def test_download_chunk_not_bytes() -> None:
    graph_http, download_http, _ = _setup_happy_path()

    class _BadChunkStream(_FakeStreamContext):
        def __enter__(self) -> MagicMock:
            response = super().__enter__()
            response.iter_bytes = lambda: iter(["not-bytes"])  # type: ignore[assignment]
            return response

    download_http.stream.return_value = _BadChunkStream(chunks=())
    with pytest.raises(
        IntegrationDependencyError,
        match="Microsoft Graph Drive download response is invalid",
    ):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [206, 302, 403, 500])
def test_download_bad_status(status_code: int) -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationDependencyError):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )


def test_download_stream_transport_exception() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    download_http.stream.side_effect = RuntimeError("stream failed")
    with pytest.raises(IntegrationDependencyError) as exc:
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_request_has_no_auth_or_graph_base() -> None:
    graph_http, download_http, reader = _setup_happy_path()
    reader.read_file_content(item=_drive_item(), max_bytes=1024)
    call = download_http.stream.call_args
    assert call.args[1] == _DOWNLOAD_URL
    assert call.kwargs["follow_redirects"] is False
    assert call.kwargs["headers"] == {"Accept": "application/octet-stream"}
    assert "Authorization" not in call.kwargs.get("headers", {})
    assert "Bearer" not in str(call.kwargs)
    assert "params" not in call.kwargs
    assert _GRAPH_BASE not in call.args[1]


# --- version / cTag ---


def test_happy_path_metadata_before_redirect_download_metadata_after() -> None:
    graph_http, download_http, reader = _setup_happy_path()
    reader.read_file_content(item=_drive_item(), max_bytes=1024)
    assert graph_http.get.call_count == 3
    assert graph_http.get.call_args_list[0].args[0] == _METADATA_PATH
    assert graph_http.get.call_args_list[1].args[0] == _CONTENT_PATH
    assert graph_http.get.call_args_list[2].args[0] == _METADATA_PATH
    assert download_http.stream.call_count == 1


def test_ctag_changed_before_download() -> None:
    graph_http = MagicMock()
    graph_http.get.return_value = _json_response(
        payload=_file_payload(c_tag='"ctag-new"'),
    )
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(MsGraphDriveContentChanged):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)
    graph_http.get.assert_called_once()


def test_ctag_changed_after_download_bytes_not_returned() -> None:
    graph_http = MagicMock()
    before = _file_payload()
    after = _file_payload(c_tag='"ctag-after"')
    redirect_response = MagicMock(status_code=302, headers={"Location": _DOWNLOAD_URL})
    graph_http.get.side_effect = [
        _json_response(payload=before),
        redirect_response,
        _json_response(payload=after),
    ]
    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(MsGraphDriveContentChanged):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(),
            max_bytes=1024,
        )


def test_item_without_ctag_rejected() -> None:
    item = MsGraphDriveItem(
        remote_id=_ITEM_ID,
        drive_id=_DRIVE_ID,
        kind=MsGraphDriveItemKind.FILE,
        name="x.pdf",
        c_tag=None,
        last_modified_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
    )
    reader = _content_reader(MagicMock(), MagicMock())
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        reader.read_file_content(item=item, max_bytes=1024)


def test_metadata_folder_rejected() -> None:
    graph_http = MagicMock()
    folder_payload = _file_payload()
    del folder_payload["file"]
    folder_payload["folder"] = {"childCount": 0}
    graph_http.get.return_value = _json_response(payload=folder_payload)
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)


def test_metadata_wrong_remote_id() -> None:
    graph_http = MagicMock()
    graph_http.get.return_value = _json_response(
        payload=_file_payload(item_id="other-id"),
    )
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(ValueError, match="unexpected Microsoft Graph drive response"):
        reader.read_file_content(item=_drive_item(), max_bytes=1024)


def test_size_before_exceeds_limit() -> None:
    graph_http = MagicMock()
    graph_http.get.return_value = _json_response(payload=_file_payload(size=500))
    reader = _content_reader(graph_http, MagicMock())
    with pytest.raises(MsGraphDriveContentTooLarge):
        reader.read_file_content(item=_drive_item(size=100), max_bytes=200)


def test_size_after_metadata_mismatch() -> None:
    graph_http = MagicMock()
    before = _file_payload(size=5)
    after = _file_payload(size=99)
    redirect_response = MagicMock(status_code=302, headers={"Location": _DOWNLOAD_URL})
    graph_http.get.side_effect = [
        _json_response(payload=before),
        redirect_response,
        _json_response(payload=after),
    ]
    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(MsGraphDriveContentChanged):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(size=5),
            max_bytes=1024,
        )


def test_input_size_mismatch_with_downloaded_bytes() -> None:
    graph_http = MagicMock()
    before = _file_payload(size=5)
    redirect_response = MagicMock(status_code=302, headers={"Location": _DOWNLOAD_URL})
    graph_http.get.side_effect = [
        _json_response(payload=before),
        redirect_response,
        _json_response(payload=before),
    ]
    download_http = MagicMock()
    download_http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(MsGraphDriveContentChanged):
        _content_reader(graph_http, download_http).read_file_content(
            item=_drive_item(size=99),
            max_bytes=1024,
        )


def test_item_size_exceeds_limit_before_any_request() -> None:
    reader = _content_reader(MagicMock(), MagicMock())
    with pytest.raises(MsGraphDriveContentTooLarge):
        reader.read_file_content(item=_drive_item(size=200), max_bytes=100)


def test_max_bytes_bool_rejected() -> None:
    reader = _content_reader(MagicMock(), MagicMock())
    with pytest.raises(IntegrationConfigurationError):
        reader.read_file_content(item=_drive_item(), max_bytes=True)  # type: ignore[arg-type]


# --- delegation ---


def test_graph_rest_client_delegates_content() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    client = _graph_client(graph_http, download_http)
    result = client.read_drive_file_content(item=_drive_item())
    assert result.data == b"hello-world"


def test_collaboration_suite_delegates_content() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    suite = _Ms365GraphCollaborationSuite(_graph_client(graph_http, download_http))
    result = suite.read_drive_file_content(item=_drive_item())
    assert result.size_bytes == len(b"hello-world")


def test_integration_delegates_content() -> None:
    graph_http, download_http, _ = _setup_happy_path()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(graph_http, download_http)),
        enabled=True,
    )
    result = integration.read_drive_file_content(item=_drive_item())
    assert result.content_revision == _CTAG


class _DriveDeltaOnlySuite(CollaborationSuite):
    def read_drive_delta_page(self, *, drive_id: str, continuation=None, limit: int = 100):
        raise NotImplementedError

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


def test_custom_suite_without_content_fails_safely() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _DriveDeltaOnlySuite(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph Drive download client is not configured",
    ):
        integration.read_drive_file_content(item=_drive_item())


def test_default_max_bytes_constant() -> None:
    assert DEFAULT_DRIVE_CONTENT_MAX_BYTES == 25 * 1024 * 1024
    assert ABSOLUTE_DRIVE_CONTENT_MAX_BYTES == 100 * 1024 * 1024
