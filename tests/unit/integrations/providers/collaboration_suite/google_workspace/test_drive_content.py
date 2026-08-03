# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationConfigurationError, IntegrationDependencyError
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GoogleDriveItem,
    GoogleDriveItemKind,
    GoogleDriveKnowledgeReader,
    GoogleDriveScope,
    GoogleDriveScopeKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive_content import (
    DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES,
    GoogleDriveContentChanged,
    GoogleDriveContentMode,
    GoogleDriveContentReader,
    GoogleDriveContentTooLarge,
    GoogleDriveContentUnavailable,
    GoogleDriveFileContent,
    GoogleDriveUnsupportedContent,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)

_USER_SCOPE = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
_SHARED_DRIVE_ID = "shared-drive-abc"
_SHARED_SCOPE = GoogleDriveScope(
    kind=GoogleDriveScopeKind.SHARED_DRIVE,
    drive_id=_SHARED_DRIVE_ID,
)
_FILE_FIELDS = (
    "id,name,mimeType,parents,driveId,webViewLink,createdTime,modifiedTime,"
    "size,md5Checksum,version,headRevisionId,trashed,"
    "shortcutDetails(targetId,targetMimeType),capabilities(canDownload)"
)


def _blob_payload(
    *,
    remote_id: str = "file-blob-1",
    drive_id: str | None = None,
    size: str = "4",
    md5: str | None = None,
    version: str = "5",
    modified: str = "2024-01-02T12:00:00Z",
    can_download: bool = True,
) -> dict[str, object]:
    data = b"test"
    checksum = md5 if md5 is not None else hashlib.md5(data, usedforsecurity=False).hexdigest()
    payload: dict[str, object] = {
        "id": remote_id,
        "name": "report.pdf",
        "mimeType": "application/pdf",
        "parents": ["parent-1"],
        "webViewLink": "https://drive.google.com/file/d/file-blob-1/view",
        "createdTime": "2024-01-01T12:00:00Z",
        "modifiedTime": modified,
        "size": size,
        "md5Checksum": checksum,
        "version": version,
        "headRevisionId": "head-rev-1",
        "trashed": False,
        "capabilities": {"canDownload": can_download},
    }
    if drive_id is not None:
        payload["driveId"] = drive_id
    return payload


def _native_payload(
    mime_type: str,
    *,
    remote_id: str = "gdoc-1",
    version: str = "3",
    modified: str = "2024-01-02T12:00:00Z",
) -> dict[str, object]:
    payload = _blob_payload(
        remote_id=remote_id,
        size="0",
        md5="",
        version=version,
        modified=modified,
    )
    payload["mimeType"] = mime_type
    payload.pop("md5Checksum", None)
    payload["size"] = "0"
    return payload


def _item_from_payload(
    payload: dict[str, object],
    *,
    scope: GoogleDriveScope = _USER_SCOPE,
) -> GoogleDriveItem:
    transport = _DualTransport(json_responses=[payload])
    return GoogleDriveKnowledgeReader(transport=transport).read_item(
        scope=scope,
        file_id=str(payload["id"]),
    )


@dataclass
class _DualTransport:
    json_responses: list[dict[str, object]] = field(default_factory=list)
    binary_responses: list[GoogleWorkspaceBinaryPayload] = field(default_factory=list)
    json_calls: list[dict[str, object]] = field(default_factory=list)
    binary_calls: list[dict[str, object]] = field(default_factory=list)
    json_exception: Exception | None = None
    binary_exception: Exception | None = None
    _json_index: int = 0

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.json_calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
            }
        )
        if self.json_exception is not None:
            raise self.json_exception
        if self._json_index < len(self.json_responses):
            response = self.json_responses[self._json_index]
            self._json_index += 1
            return response
        return self.json_responses[-1]

    def get_bytes(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None,
        expected_content_type: str,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        self.binary_calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "expected_content_type": expected_content_type,
                "max_bytes": max_bytes,
                "range_limited": range_limited,
            }
        )
        if self.binary_exception is not None:
            raise self.binary_exception
        return self.binary_responses.pop(0)


class _JsonOnlyTransport:
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        return {}


def test_content_reader_rejects_json_only_transport() -> None:
    with pytest.raises(IntegrationConfigurationError, match="binary content"):
        GoogleDriveContentReader(transport=_JsonOnlyTransport())


def test_content_reader_accepts_dual_capability_transport() -> None:
    transport = _DualTransport()
    reader = GoogleDriveContentReader(transport=transport)
    assert reader._transport is transport
    assert reader._metadata_reader._transport is transport


def test_blob_success_exact_sequence_and_request() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    data = b"test"
    transport = _DualTransport(
        json_responses=[payload, payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=data, content_type="application/pdf")],
    )
    reader = GoogleDriveContentReader(transport=transport)
    result = reader.read_drive_file_content(item=item)
    assert len(transport.json_calls) == 2
    assert len(transport.binary_calls) == 1
    assert transport.json_calls[0]["relative_path"] == transport.json_calls[1]["relative_path"]
    binary = transport.binary_calls[0]
    assert binary["relative_path"] == "/files/file-blob-1"
    assert binary["params"] == {"alt": "media", "supportsAllDrives": True}
    assert binary["expected_content_type"] == "application/pdf"
    assert binary["range_limited"] is True
    assert binary["max_bytes"] == DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES
    assert result.mode is GoogleDriveContentMode.BLOB
    assert result.size_bytes == len(data)
    assert result.content_hash == hashlib.sha256(data).hexdigest()
    assert "test" not in repr(result)


def test_blob_shared_drive_item() -> None:
    payload = _blob_payload(drive_id=_SHARED_DRIVE_ID)
    item = _item_from_payload(payload, scope=_SHARED_SCOPE)
    transport = _DualTransport(
        json_responses=[payload, payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="application/pdf")],
    )
    result = GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert result.item.scope == _SHARED_SCOPE


def test_blob_zero_byte_file() -> None:
    payload = _blob_payload(size="0", md5="d41d8cd98f00b204e9800998ecf8427e")
    item = _item_from_payload(payload)
    transport = _DualTransport(
        json_responses=[payload, payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"", content_type="application/pdf")],
    )
    result = GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert result.size_bytes == 0


@pytest.mark.parametrize(
    ("native_mime", "export_mime"),
    [
        (
            "application/vnd.google-apps.document",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "application/vnd.google-apps.spreadsheet",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            "application/vnd.google-apps.presentation",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        ("application/vnd.google-apps.drawing", "application/pdf"),
        ("application/vnd.google-apps.script", "application/vnd.google-apps.script+json"),
    ],
)
def test_native_export_success(native_mime: str, export_mime: str) -> None:
    payload = _native_payload(native_mime)
    item = _item_from_payload(payload)
    data = b"export-data"
    transport = _DualTransport(
        json_responses=[payload, payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=data, content_type=export_mime)],
    )
    reader = GoogleDriveContentReader(transport=transport)
    result = reader.read_drive_file_content(item=item)
    binary = transport.binary_calls[0]
    assert binary["relative_path"] == f"/files/{payload['id']}/export"
    assert binary["params"] == {"mimeType": export_mime}
    assert binary["range_limited"] is False
    assert "supportsAllDrives" not in binary["params"]
    assert "alt" not in binary["params"]
    assert binary["max_bytes"] <= GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES
    assert result.mode is GoogleDriveContentMode.EXPORT
    assert result.content_mime_type == export_mime


@pytest.mark.parametrize(
    "mime_type",
    [
        "application/vnd.google-apps.folder",
        "application/vnd.google-apps.shortcut",
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.site",
        "application/vnd.google-apps.map",
        "application/vnd.google-apps.fusiontable",
        "application/vnd.google-apps.jam",
        "application/vnd.google-apps.vid",
        "application/vnd.google-apps.unknown",
    ],
)
def test_unsupported_content_rejected_before_binary(mime_type: str) -> None:
    if mime_type == "application/vnd.google-apps.folder":
        payload = _blob_payload()
        payload.update(
            {
                "mimeType": mime_type,
            }
        )
        payload.pop("size", None)
        payload.pop("md5Checksum", None)
        payload.pop("headRevisionId", None)
    elif mime_type == "application/vnd.google-apps.shortcut":
        payload = _blob_payload()
        payload.update(
            {
                "mimeType": mime_type,
                "shortcutDetails": {"targetId": "target-1", "targetMimeType": "application/pdf"},
            }
        )
        payload.pop("size", None)
        payload.pop("md5Checksum", None)
        payload.pop("headRevisionId", None)
    else:
        payload = _native_payload(mime_type)
    item = _item_from_payload(payload)
    transport = _DualTransport()
    with pytest.raises((GoogleDriveUnsupportedContent, IntegrationConfigurationError)):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert transport.binary_calls == []


def test_caller_can_download_false_rejected() -> None:
    payload = _blob_payload(can_download=False)
    item = _item_from_payload(payload)
    transport = _DualTransport()
    with pytest.raises(GoogleDriveContentUnavailable):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert transport.json_calls == []


def test_metadata_before_can_download_false() -> None:
    payload = _blob_payload(can_download=True)
    item = _item_from_payload(payload)
    before = dict(payload)
    before["capabilities"] = {"canDownload": False}
    transport = _DualTransport(json_responses=[before])
    with pytest.raises(GoogleDriveContentUnavailable):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert transport.binary_calls == []


def test_declared_size_above_limit_no_network() -> None:
    payload = _blob_payload(size=str(DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES + 1))
    item = _item_from_payload(payload)
    transport = _DualTransport()
    with pytest.raises(GoogleDriveContentTooLarge):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert transport.json_calls == []


def test_revision_fence_stale_caller_version() -> None:
    payload = _blob_payload(version="5")
    stale = _item_from_payload(payload)
    changed = dict(payload)
    changed["version"] = "6"
    transport = _DualTransport(json_responses=[changed])
    with pytest.raises(GoogleDriveContentChanged):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=stale)


def test_revision_fence_metadata_after_version_change() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    after = dict(payload)
    after["version"] = "99"
    transport = _DualTransport(
        json_responses=[payload, after],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="application/pdf")],
    )
    with pytest.raises(GoogleDriveContentChanged):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)


def test_md5_mismatch_raises_content_changed() -> None:
    payload = _blob_payload(md5="00000000000000000000000000000000")
    item = _item_from_payload(payload)
    transport = _DualTransport(
        json_responses=[payload, payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="application/pdf")],
    )
    with pytest.raises(GoogleDriveContentChanged):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)


def test_binary_payload_too_large_converted() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    transport = _DualTransport(
        json_responses=[payload],
        binary_exception=GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
            status_code=200,
            retry_after_seconds=None,
            safe_reason="payload_too_large",
            attempts=1,
        ),
    )
    with pytest.raises(GoogleDriveContentTooLarge):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)


@pytest.mark.parametrize(
    "kind",
    [
        GoogleWorkspaceErrorKind.AUTHENTICATION,
        GoogleWorkspaceErrorKind.AUTHORIZATION,
        GoogleWorkspaceErrorKind.NOT_FOUND,
        GoogleWorkspaceErrorKind.RATE_LIMITED,
        GoogleWorkspaceErrorKind.TEMPORARY,
        GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
        GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT,
    ],
)
def test_transport_errors_propagate_unchanged(kind: GoogleWorkspaceErrorKind) -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    api_error = GoogleWorkspaceApiError(
        kind=kind,
        status_code=500,
        retry_after_seconds=None,
        safe_reason="safe",
        attempts=1,
    )
    transport = _DualTransport(json_responses=[payload], binary_exception=api_error)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert exc_info.value is api_error


def test_safe_errors_do_not_expose_sensitive_values() -> None:
    payload = _blob_payload(remote_id="secret-file-id")
    item = _item_from_payload(payload)
    changed = dict(payload)
    changed["version"] = "99"
    transport = _DualTransport(
        json_responses=[payload, changed],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="application/pdf")],
    )
    with pytest.raises(GoogleDriveContentChanged) as exc_info:
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    message = str(exc_info.value)
    for forbidden in ("secret-file-id", "application/pdf", "head-rev-1", "test", "safe"):
        assert forbidden not in message


def test_malformed_content_model_rejected() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    with pytest.raises((ValidationError, IntegrationConfigurationError)):
        GoogleDriveFileContent(
            item=item,
            mode=GoogleDriveContentMode.BLOB,
            content_mime_type="application/pdf",
            data=b"abc",
            size_bytes=99,
            content_hash=hashlib.sha256(b"abc").hexdigest(),
        )


def test_valid_item_is_copied_not_identical() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    transport = _DualTransport(
        json_responses=[payload, payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="application/pdf")],
    )
    result = GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert result.item == item
    assert result.item is not item


@pytest.mark.parametrize(
    ("construct_kwargs",),
    [
        ({"kind": GoogleDriveItemKind.BLOB, "mime_type": "application/vnd.google-apps.document"},),
        ({"scope": {"kind": GoogleDriveScopeKind.SHARED_DRIVE}},),
        ({"can_download": "yes"},),
        ({"version": None},),
    ],
)
def test_model_construct_item_rejected_before_network(construct_kwargs: dict[str, object]) -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    snapshot = item.model_dump(mode="python")
    snapshot.update(construct_kwargs)
    bad_item = GoogleDriveItem.model_construct(**snapshot)
    transport = _DualTransport()
    with pytest.raises(IntegrationConfigurationError, match="invalid Google Drive content item"):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=bad_item)
    assert transport.json_calls == []
    assert transport.binary_calls == []


def test_item_subclass_rejected_before_network() -> None:
    class _SubclassItem(GoogleDriveItem):
        pass

    payload = _blob_payload()
    item = _item_from_payload(payload)
    snapshot = item.model_dump(mode="python")
    snapshot["scope"] = GoogleDriveScope(**snapshot["scope"])
    subclass_item = _SubclassItem(**snapshot)
    transport = _DualTransport()
    with pytest.raises(IntegrationConfigurationError, match="invalid Google Drive content item"):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=subclass_item)
    assert transport.json_calls == []
    assert transport.binary_calls == []


def test_file_content_rejects_model_construct_item() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    snapshot = item.model_dump(mode="python")
    snapshot["kind"] = GoogleDriveItemKind.NATIVE_DOCUMENT
    bad_item = GoogleDriveItem.model_construct(**snapshot)
    with pytest.raises((ValidationError, IntegrationConfigurationError)):
        GoogleDriveFileContent(
            item=bad_item,
            mode=GoogleDriveContentMode.BLOB,
            content_mime_type="application/pdf",
            data=b"test",
            size_bytes=4,
            content_hash=hashlib.sha256(b"test").hexdigest(),
        )


def test_file_content_rejects_item_subclass() -> None:
    class _SubclassItem(GoogleDriveItem):
        pass

    payload = _blob_payload()
    item = _item_from_payload(payload)
    snapshot = item.model_dump(mode="python")
    snapshot["scope"] = GoogleDriveScope(**snapshot["scope"])
    subclass_item = _SubclassItem(**snapshot)
    with pytest.raises((ValidationError, IntegrationConfigurationError)):
        GoogleDriveFileContent(
            item=subclass_item,
            mode=GoogleDriveContentMode.BLOB,
            content_mime_type="application/pdf",
            data=b"test",
            size_bytes=4,
            content_hash=hashlib.sha256(b"test").hexdigest(),
        )


@pytest.mark.parametrize(
    "content_mime_type",
    ["application/pdf; charset=utf-8", "text/*", "application/pdf,"],
)
def test_file_content_rejects_malformed_content_mime(content_mime_type: str) -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    with pytest.raises((ValidationError, IntegrationConfigurationError)):
        GoogleDriveFileContent(
            item=item,
            mode=GoogleDriveContentMode.BLOB,
            content_mime_type=content_mime_type,
            data=b"test",
            size_bytes=4,
            content_hash=hashlib.sha256(b"test").hexdigest(),
        )

def test_file_content_uppercase_content_mime_canonicalized() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    result = GoogleDriveFileContent(
        item=item,
        mode=GoogleDriveContentMode.BLOB,
        content_mime_type="APPLICATION/PDF",
        data=b"test",
        size_bytes=4,
        content_hash=hashlib.sha256(b"test").hexdigest(),
    )
    assert result.content_mime_type == "application/pdf"


def test_file_content_valid_item_stored_as_copy() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    result = GoogleDriveFileContent(
        item=item,
        mode=GoogleDriveContentMode.BLOB,
        content_mime_type="application/pdf",
        data=b"test",
        size_bytes=4,
        content_hash=hashlib.sha256(b"test").hexdigest(),
    )
    assert result.item == item
    assert result.item is not item


    payload = _blob_payload()
    item = _item_from_payload(payload)
    result = GoogleDriveFileContent(
        item=item,
        mode=GoogleDriveContentMode.BLOB,
        content_mime_type="application/pdf",
        data=b"test",
        size_bytes=4,
        content_hash=hashlib.sha256(b"test").hexdigest(),
    )
    assert result.item == item
    assert result.item is not item


class _WrongBinaryReturnTransport(_DualTransport):
    def get_bytes(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None,
        expected_content_type: str,
        max_bytes: int,
        range_limited: bool,
    ) -> object:
        self.binary_calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "expected_content_type": expected_content_type,
                "max_bytes": max_bytes,
                "range_limited": range_limited,
            }
        )
        return "not-a-payload"


def test_injected_wrong_binary_return_type() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    transport = _WrongBinaryReturnTransport(
        json_responses=[payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="application/pdf")],
    )
    with pytest.raises(IntegrationDependencyError, match="invalid Google Drive binary content result"):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    assert len(transport.json_calls) == 1
    assert transport.binary_calls


def test_injected_wrong_payload_content_type() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    transport = _DualTransport(
        json_responses=[payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=b"test", content_type="text/plain")],
    )
    with pytest.raises(IntegrationDependencyError, match="invalid Google Drive binary content result") as exc_info:
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)
    message = str(exc_info.value)
    assert "text/plain" not in message
    assert "test" not in message


def test_injected_malformed_binary_payload() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    bad_payload = object.__new__(GoogleWorkspaceBinaryPayload)
    object.__setattr__(bad_payload, "data", b"test")
    object.__setattr__(bad_payload, "content_type", "text/plain; charset=utf-8")
    transport = _DualTransport(json_responses=[payload], binary_responses=[bad_payload])
    with pytest.raises(IntegrationDependencyError, match="invalid Google Drive binary content result"):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)


def test_injected_non_bytes_binary_payload() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    bad_payload = object.__new__(GoogleWorkspaceBinaryPayload)
    object.__setattr__(bad_payload, "data", "not-bytes")
    object.__setattr__(bad_payload, "content_type", "application/pdf")
    transport = _DualTransport(json_responses=[payload], binary_responses=[bad_payload])
    with pytest.raises(IntegrationDependencyError, match="invalid Google Drive binary content result"):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(item=item)


def test_native_export_effective_limit_enforced() -> None:
    native_payload = _native_payload("application/vnd.google-apps.document")
    item = _item_from_payload(native_payload)
    oversized = b"x" * (GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES + 1)
    transport = _DualTransport(
        json_responses=[native_payload],
        binary_responses=[
            GoogleWorkspaceBinaryPayload(
                data=oversized,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        ],
    )
    with pytest.raises(GoogleDriveContentTooLarge):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(
            item=item,
            max_bytes=DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
        )
    assert transport.binary_calls[0]["max_bytes"] == GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES
    assert len(transport.json_calls) == 1


def test_blob_effective_limit_enforced() -> None:
    payload = _blob_payload()
    item = _item_from_payload(payload)
    max_bytes = 1024
    oversized = b"x" * (max_bytes + 1)
    transport = _DualTransport(
        json_responses=[payload],
        binary_responses=[GoogleWorkspaceBinaryPayload(data=oversized, content_type="application/pdf")],
    )
    with pytest.raises(GoogleDriveContentTooLarge):
        GoogleDriveContentReader(transport=transport).read_drive_file_content(
            item=item,
            max_bytes=max_bytes,
        )
    assert transport.binary_calls[0]["max_bytes"] == max_bytes
    assert len(transport.json_calls) == 1
