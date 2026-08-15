# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
    GoogleDriveChange,
    GoogleDriveChangePage,
    GoogleDriveItem,
    GoogleDriveItemKind,
    GoogleDriveItemPage,
    GoogleDriveKnowledgeReader,
    GoogleDriveScope,
    GoogleDriveScopeKind,
    GoogleDriveSharedDrive,
    GoogleDriveSharedDrivePage,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
    GoogleWorkspacePageToken,
)

_FILE_FIELDS = (
    "id,name,mimeType,parents,driveId,webViewLink,createdTime,modifiedTime,"
    "size,md5Checksum,version,headRevisionId,trashed,"
    "shortcutDetails(targetId,targetMimeType),capabilities(canDownload)"
)
_INVENTORY_FIELDS = f"nextPageToken,incompleteSearch,files({_FILE_FIELDS})"
_SHARED_DRIVE_LIST_FIELDS = "nextPageToken,drives(id,name,createdTime,hidden)"
_CHANGE_FIELDS = (
    f"nextPageToken,newStartPageToken,changes("
    f"changeType,removed,fileId,time,driveId,file({_FILE_FIELDS}))"
)

_USER_SCOPE = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
_SHARED_DRIVE_ID = "shared-drive-abc"
_SHARED_SCOPE = GoogleDriveScope(
    kind=GoogleDriveScopeKind.SHARED_DRIVE,
    drive_id=_SHARED_DRIVE_ID,
)


def _capabilities(can_download: bool = True) -> dict[str, object]:
    return {"canDownload": can_download}


def _blob_payload(
    *,
    remote_id: str = "file-blob-1",
    drive_id: str | None = None,
    parents: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": remote_id,
        "name": "report.pdf",
        "mimeType": "application/pdf",
        "parents": parents or ["parent-1"],
        "webViewLink": "https://drive.google.com/file/d/file-blob-1/view",
        "createdTime": "2024-01-01T12:00:00Z",
        "modifiedTime": "2024-01-02T12:00:00Z",
        "size": "1024",
        "md5Checksum": "d41d8cd98f00b204e9800998ecf8427e",
        "version": "5",
        "headRevisionId": "head-rev-1",
        "trashed": False,
        "capabilities": _capabilities(True),
    }
    if drive_id is not None:
        payload["driveId"] = drive_id
    return payload


def _folder_payload(**kwargs: object) -> dict[str, object]:
    base = _blob_payload(remote_id="folder-1", parents=["root-parent"])
    base.update(
        {
            "name": "Folder One",
            "mimeType": "application/vnd.google-apps.folder",
            "size": None,
            "md5Checksum": None,
            "headRevisionId": None,
            "webViewLink": None,
        }
    )
    base.pop("size", None)
    base.pop("md5Checksum", None)
    base.pop("headRevisionId", None)
    base.pop("webViewLink", None)
    base.update(kwargs)
    return base


def _native_doc_payload() -> dict[str, object]:
    payload = _blob_payload(remote_id="gdoc-1")
    payload.update(
        {
            "name": "Notes",
            "mimeType": "application/vnd.google-apps.document",
            "size": "0",
            "md5Checksum": None,
        }
    )
    payload.pop("md5Checksum", None)
    return payload


def _shortcut_payload() -> dict[str, object]:
    payload = _blob_payload(remote_id="shortcut-1")
    payload.update(
        {
            "name": "Shortcut",
            "mimeType": "application/vnd.google-apps.shortcut",
            "size": None,
            "md5Checksum": None,
            "shortcutDetails": {
                "targetId": "target-file-1",
                "targetMimeType": "application/pdf",
            },
        }
    )
    payload.pop("size", None)
    payload.pop("md5Checksum", None)
    return payload


@dataclass
class _RecordingTransport:
    responses: list[dict[str, object]] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)
    exception: Exception | None = None

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            }
        )
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            return {}
        return self.responses.pop(0)


def test_scope_valid_user() -> None:
    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    assert scope.kind is GoogleDriveScopeKind.USER
    assert scope.drive_id is None


def test_scope_valid_shared_drive() -> None:
    scope = GoogleDriveScope(
        kind=GoogleDriveScopeKind.SHARED_DRIVE,
        drive_id="  drive-xyz  ",
    )
    assert scope.drive_id == "drive-xyz"


def test_scope_user_with_drive_id_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveScope(kind=GoogleDriveScopeKind.USER, drive_id="drive-1")


def test_scope_shared_drive_without_drive_id_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveScope(kind=GoogleDriveScopeKind.SHARED_DRIVE)


@pytest.mark.parametrize(
    "drive_id",
    [
        "",
        "   ",
        "x" * 1025,
        "drive\x00id",
        123,
        True,
    ],
)
def test_scope_drive_id_validation_rejects_invalid(drive_id: object) -> None:
    with pytest.raises(ValidationError):
        GoogleDriveScope(kind=GoogleDriveScopeKind.SHARED_DRIVE, drive_id=drive_id)


def test_shared_drive_page_valid() -> None:
    page = GoogleDriveSharedDrivePage(
        items=(
            GoogleDriveSharedDrive(
                remote_id="drive-1",
                name="Team Drive",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                hidden=False,
            ),
        ),
    )
    assert len(page.items) == 1


def test_shared_drive_page_next_token() -> None:
    token = GoogleWorkspacePageToken(value="next-shared")
    page = GoogleDriveSharedDrivePage(items=(), next_page_token=token)
    assert "next-shared" not in repr(page)


def test_shared_drive_page_preserves_ordering() -> None:
    items = tuple(
        GoogleDriveSharedDrive(
            remote_id=f"drive-{index}",
            name=f"Drive {index}",
            created_at=datetime(2024, 1, index + 1, tzinfo=timezone.utc),
            hidden=False,
        )
        for index in range(3)
    )
    page = GoogleDriveSharedDrivePage(items=items)
    assert [item.remote_id for item in page.items] == ["drive-0", "drive-1", "drive-2"]


def test_shared_drive_page_duplicate_ids_rejected() -> None:
    drive = GoogleDriveSharedDrive(
        remote_id="dup",
        name="One",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        hidden=False,
    )
    with pytest.raises(ValidationError):
        GoogleDriveSharedDrivePage(items=(drive, drive))


def test_shared_drive_invalid_name_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveSharedDrive(
            remote_id="drive-1",
            name="",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            hidden=False,
        )


def test_shared_drive_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveSharedDrive(
            remote_id="drive-1",
            name="Drive",
            created_at=datetime(2024, 1, 1),
            hidden=False,
        )


def test_shared_drive_malformed_hidden_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveSharedDrive(
            remote_id="drive-1",
            name="Drive",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            hidden="false",
        )


def test_item_blob_valid() -> None:
    item = GoogleDriveItem(
        remote_id="file-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=("parent-1",),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        size_bytes=1024,
        md5_checksum="checksum",
        version=3,
        head_revision_id="rev-1",
        web_view_link="https://drive.google.com/file/d/file-1/view",
        can_download=True,
    )
    assert item.kind is GoogleDriveItemKind.BLOB
    assert "checksum" not in repr(item)
    assert "drive.google.com" not in repr(item)


def test_item_folder_valid() -> None:
    item = GoogleDriveItem(
        remote_id="folder-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.FOLDER,
        name="Folder",
        mime_type="application/vnd.google-apps.folder",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=False,
    )
    assert item.size_bytes is None


def test_item_native_document_valid() -> None:
    item = GoogleDriveItem(
        remote_id="gdoc-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.NATIVE_DOCUMENT,
        name="Doc",
        mime_type="application/vnd.google-apps.document",
        parent_ids=("parent-1",),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        size_bytes=0,
        version=2,
        can_download=True,
    )
    assert item.kind is GoogleDriveItemKind.NATIVE_DOCUMENT


def test_item_shortcut_valid() -> None:
    item = GoogleDriveItem(
        remote_id="shortcut-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.SHORTCUT,
        name="Shortcut",
        mime_type="application/vnd.google-apps.shortcut",
        parent_ids=("parent-1",),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
        shortcut_target_id="target-1",
        shortcut_target_mime_type="application/pdf",
    )
    assert item.shortcut_target_id == "target-1"


def test_item_size_and_version_decimal_string_parsing() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "files": [_blob_payload()],
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_items_page(scope=_USER_SCOPE)
    item = page.items[0]
    assert item.size_bytes == 1024
    assert item.version == 5


@pytest.mark.parametrize("bad_size", ["-1", "1.5", " 10", "10 ", "+1", True, 1.0])
def test_item_invalid_numeric_strings_rejected(bad_size: object) -> None:
    payload = _blob_payload()
    payload["size"] = bad_size
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match="unexpected"):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_negative_version_rejected() -> None:
    payload = _blob_payload()
    payload["version"] = "-1"
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_boolean_version_rejected() -> None:
    payload = _blob_payload()
    payload["version"] = True
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_invalid_mime_type_rejected() -> None:
    payload = _blob_payload()
    payload["mimeType"] = ""
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_duplicate_parent_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=_USER_SCOPE,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=("parent-1", "parent-1"),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
        )


def test_item_mismatched_shared_drive_scope_rejected() -> None:
    payload = _blob_payload(drive_id="other-drive")
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_SHARED_SCOPE)


def test_item_missing_shortcut_target_rejected() -> None:
    payload = _shortcut_payload()
    payload.pop("shortcutDetails")
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_shortcut_fields_on_non_shortcut_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=_USER_SCOPE,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=(),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
            shortcut_target_id="target-1",
            shortcut_target_mime_type="application/pdf",
        )


def test_item_invalid_capability_shape_rejected() -> None:
    payload = _blob_payload()
    payload["capabilities"] = {"canDownload": "yes"}
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_trashed_true_rejected() -> None:
    payload = _blob_payload()
    payload["trashed"] = True
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_USER_SCOPE)


def test_item_naive_dates_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=_USER_SCOPE,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=(),
            created_at=datetime(2024, 1, 1),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
        )


def test_item_page_duplicate_ids_rejected() -> None:
    item = GoogleDriveItem(
        remote_id="dup",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="a",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    with pytest.raises(ValidationError):
        GoogleDriveItemPage(items=(item, item))


def test_change_non_removed_file_change() -> None:
    item = GoogleDriveItem(
        remote_id="file-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    change = GoogleDriveChange(
        file_id="file-1",
        scope=_USER_SCOPE,
        removed=False,
        changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        item=item,
    )
    assert change.item is item


def test_change_removed_tombstone_without_file() -> None:
    change = GoogleDriveChange(
        file_id="file-removed",
        scope=_USER_SCOPE,
        removed=True,
        changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    assert change.item is None


def test_change_ignored_drive_change_type() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "drive",
                        "removed": False,
                        "fileId": "ignored",
                        "time": "2024-01-03T12:00:00Z",
                    },
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-removed",
                        "time": "2024-01-03T12:00:00Z",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_changes_page(
        scope=_USER_SCOPE,
        page_token=GoogleWorkspacePageToken(value="start"),
    )
    assert len(page.changes) == 1
    assert page.changes[0].file_id == "file-removed"


def test_change_unknown_change_type_rejected() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "unknown",
                        "removed": False,
                        "fileId": "file-1",
                        "time": "2024-01-03T12:00:00Z",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_USER_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_change_item_file_id_mismatch_rejected() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": False,
                        "fileId": "file-1",
                        "time": "2024-01-03T12:00:00Z",
                        "file": _blob_payload(remote_id="other-id"),
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_USER_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_change_scope_mismatch_rejected() -> None:
    item = GoogleDriveItem(
        remote_id="file-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    with pytest.raises(ValidationError):
        GoogleDriveChange(
            file_id="file-1",
            scope=_SHARED_SCOPE,
            removed=False,
            changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
            item=item,
        )


def test_change_page_valid_next_page_token() -> None:
    token = GoogleWorkspacePageToken(value="next-changes")
    page = GoogleDriveChangePage(
        changes=(),
        next_page_token=token,
    )
    assert "next-changes" not in repr(page)


def test_change_page_valid_new_start_token() -> None:
    token = GoogleWorkspacePageToken(value="new-start")
    page = GoogleDriveChangePage(
        changes=(),
        new_start_page_token=token,
    )
    assert "new-start" not in repr(page)


def test_change_page_both_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveChangePage(
            changes=(),
            next_page_token=GoogleWorkspacePageToken(value="next"),
            new_start_page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_change_page_neither_token_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveChangePage(changes=())


def test_change_malformed_time_rejected() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-1",
                        "time": "not-a-date",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_USER_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_change_malformed_file_payload_rejected() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": False,
                        "fileId": "file-1",
                        "time": "2024-01-03T12:00:00Z",
                        "file": "not-a-dict",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_USER_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_reader_construction_no_network_call() -> None:
    transport = _RecordingTransport()
    GoogleDriveKnowledgeReader(transport=transport)
    assert transport.calls == []


def test_reader_invalid_transport_rejected() -> None:
    with pytest.raises(IntegrationConfigurationError):
        GoogleDriveKnowledgeReader(transport=object())  # type: ignore[arg-type]


def test_list_shared_drives_exact_request() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "drives": [
                    {
                        "id": "drive-1",
                        "name": "Team",
                        "createdTime": "2024-01-01T00:00:00Z",
                        "hidden": False,
                    },
                ],
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.list_shared_drives_page(limit=50)
    call = transport.calls[0]
    assert call["source_kind"] is GoogleWorkspaceSourceKind.DRIVE
    assert call["relative_path"] == "/drives"
    assert call["params"]["pageSize"] == 50
    assert call["params"]["fields"] == _SHARED_DRIVE_LIST_FIELDS
    assert "pageToken" not in call["params"]


def test_list_shared_drives_with_page_token() -> None:
    transport = _RecordingTransport(responses=[{"drives": []}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    token = GoogleWorkspacePageToken(value="shared-next")
    reader.list_shared_drives_page(page_token=token)
    assert transport.calls[0]["params"]["pageToken"] == "shared-next"


def test_user_inventory_exact_request() -> None:
    transport = _RecordingTransport(responses=[{"files": [_blob_payload()]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.read_items_page(scope=_USER_SCOPE, limit=100)
    params = transport.calls[0]["params"]
    assert transport.calls[0]["relative_path"] == "/files"
    assert params["corpora"] == "user"
    assert params["includeItemsFromAllDrives"] is False
    assert params["spaces"] == "drive"
    assert params["q"] == "trashed = false"
    assert params["supportsAllDrives"] is True
    assert params["fields"] == _INVENTORY_FIELDS
    assert "driveId" not in params


def test_shared_drive_inventory_exact_request() -> None:
    transport = _RecordingTransport(
        responses=[{"files": [_blob_payload(drive_id=_SHARED_DRIVE_ID)]}]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.read_items_page(scope=_SHARED_SCOPE)
    params = transport.calls[0]["params"]
    assert params["corpora"] == "drive"
    assert params["driveId"] == _SHARED_DRIVE_ID
    assert params["includeItemsFromAllDrives"] is True
    assert params["supportsAllDrives"] is True


def test_read_item_exact_request() -> None:
    file_id = "file+special=id"
    transport = _RecordingTransport(responses=[_blob_payload(remote_id=file_id)])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.read_item(scope=_USER_SCOPE, file_id=file_id)
    call = transport.calls[0]
    assert call["relative_path"] == "/files/file%2Bspecial%3Did"
    assert call["params"]["supportsAllDrives"] is True
    assert call["params"]["fields"] == _FILE_FIELDS


def test_read_item_slash_id_rejected() -> None:
    transport = _RecordingTransport()
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationConfigurationError, match="identifier"):
        reader.read_item(scope=_USER_SCOPE, file_id="path/segment")


def test_start_page_token_user_scope() -> None:
    transport = _RecordingTransport(responses=[{"startPageToken": "start-1"}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    token = reader.read_start_page_token(scope=_USER_SCOPE)
    assert token.value == "start-1"
    params = transport.calls[0]["params"]
    assert transport.calls[0]["relative_path"] == "/changes/startPageToken"
    assert "driveId" not in params


def test_start_page_token_shared_drive_scope() -> None:
    transport = _RecordingTransport(responses=[{"startPageToken": "start-shared"}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.read_start_page_token(scope=_SHARED_SCOPE)
    assert transport.calls[0]["params"]["driveId"] == _SHARED_DRIVE_ID


def test_changes_page_exact_request() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page_token = GoogleWorkspacePageToken(value="change-token")
    reader.read_changes_page(scope=_SHARED_SCOPE, page_token=page_token, limit=150)
    params = transport.calls[0]["params"]
    assert transport.calls[0]["relative_path"] == "/changes"
    assert params["pageToken"] == "change-token"
    assert params["pageSize"] == 150
    assert params["spaces"] == "drive"
    assert params["includeRemoved"] is True
    assert params["includeItemsFromAllDrives"] is True
    assert params["supportsAllDrives"] is True
    assert params["driveId"] == _SHARED_DRIVE_ID
    assert params["fields"] == _CHANGE_FIELDS


def test_caller_inputs_not_mutated() -> None:
    scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    token = GoogleWorkspacePageToken(value="page-token-value")
    transport = _RecordingTransport(responses=[{"files": []}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.read_items_page(scope=scope, page_token=token, limit=10)
    assert scope.kind is GoogleDriveScopeKind.USER
    assert token.value == "page-token-value"


def test_transport_exceptions_propagate() -> None:
    api_error = GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.NOT_FOUND,
        status_code=404,
        retry_after_seconds=None,
        safe_reason="not_found",
        attempts=1,
    )
    transport = _RecordingTransport(exception=api_error)
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        reader.read_items_page(scope=_USER_SCOPE)
    assert exc_info.value is api_error


def test_malformed_provider_response_normalized() -> None:
    transport = _RecordingTransport(responses=[{"files": "not-a-list"}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match="unexpected") as exc_info:
        reader.read_items_page(scope=_USER_SCOPE)
    assert "not-a-list" not in str(exc_info.value)


def test_incomplete_search_true_fails_closed() -> None:
    transport = _RecordingTransport(
        responses=[{"files": [], "incompleteSearch": True}]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match="incomplete"):
        reader.read_items_page(scope=_USER_SCOPE)


def test_invalid_page_limit_rejected() -> None:
    transport = _RecordingTransport()
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationConfigurationError, match="page limit"):
        reader.read_items_page(scope=_USER_SCOPE, limit=0)
    with pytest.raises(IntegrationConfigurationError, match="page limit"):
        reader.read_items_page(scope=_USER_SCOPE, limit=201)


def test_invalid_page_token_type_rejected() -> None:
    transport = _RecordingTransport()
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationConfigurationError, match="page token"):
        reader.read_items_page(scope=_USER_SCOPE, page_token="raw-token")  # type: ignore[arg-type]


def test_googl_drive_source_kind_constant() -> None:
    assert GOOGLE_DRIVE_SOURCE_KIND == "drive"


def test_parse_folder_native_shortcut_kinds() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "files": [
                    _folder_payload(),
                    _native_doc_payload(),
                    _shortcut_payload(),
                ],
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_items_page(scope=_USER_SCOPE)
    kinds = [item.kind for item in page.items]
    assert kinds == [
        GoogleDriveItemKind.FOLDER,
        GoogleDriveItemKind.NATIVE_DOCUMENT,
        GoogleDriveItemKind.SHORTCUT,
    ]


# --- Scope ownership: USER items ---


def test_user_item_without_drive_id_passes() -> None:
    transport = _RecordingTransport(responses=[{"files": [_blob_payload()]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_items_page(scope=_USER_SCOPE)
    assert page.items[0].drive_id is None


def test_user_item_with_drive_id_fails() -> None:
    payload = _blob_payload(drive_id="unexpected-drive")
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match="unexpected"):
        reader.read_items_page(scope=_USER_SCOPE)


def test_user_item_direct_model_with_drive_id_fails() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=_USER_SCOPE,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=(),
            drive_id="unexpected-drive",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
        )


# --- Scope ownership: SHARED_DRIVE items ---


def test_shared_drive_item_matching_drive_id_passes() -> None:
    payload = _blob_payload(drive_id=_SHARED_DRIVE_ID)
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_items_page(scope=_SHARED_SCOPE)
    assert page.items[0].drive_id == _SHARED_DRIVE_ID


def test_shared_drive_item_missing_drive_id_fails() -> None:
    transport = _RecordingTransport(responses=[{"files": [_blob_payload()]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_SHARED_SCOPE)


def test_shared_drive_item_different_drive_id_fails() -> None:
    payload = _blob_payload(drive_id="other-drive")
    transport = _RecordingTransport(responses=[{"files": [payload]}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_items_page(scope=_SHARED_SCOPE)


def test_shared_drive_item_direct_model_without_drive_id_fails() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=_SHARED_SCOPE,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=(),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
        )


def test_shared_drive_read_item_enforces_ownership() -> None:
    payload = _blob_payload(drive_id="other-drive")
    transport = _RecordingTransport(responses=[payload])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_item(scope=_SHARED_SCOPE, file_id="file-blob-1")


def test_shared_drive_change_item_enforces_ownership() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": False,
                        "fileId": "file-blob-1",
                        "time": "2024-01-03T12:00:00Z",
                        "driveId": _SHARED_DRIVE_ID,
                        "file": _blob_payload(drive_id=_SHARED_DRIVE_ID),
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_changes_page(
        scope=_SHARED_SCOPE,
        page_token=GoogleWorkspacePageToken(value="start"),
    )
    assert page.changes[0].item is not None
    assert page.changes[0].item.drive_id == _SHARED_DRIVE_ID


# --- Change ownership: USER ---


def test_user_changes_include_items_from_all_drives_false() -> None:
    transport = _RecordingTransport(
        responses=[{"changes": [], "newStartPageToken": "checkpoint"}]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    reader.read_changes_page(
        scope=_USER_SCOPE,
        page_token=GoogleWorkspacePageToken(value="start"),
    )
    params = transport.calls[0]["params"]
    assert params["includeItemsFromAllDrives"] is False
    assert "driveId" not in params


def test_user_change_with_drive_id_fails() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-1",
                        "time": "2024-01-03T12:00:00Z",
                        "driveId": "some-drive",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_USER_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_user_change_item_with_drive_id_fails() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": False,
                        "fileId": "file-blob-1",
                        "time": "2024-01-03T12:00:00Z",
                        "file": _blob_payload(drive_id="some-drive"),
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_USER_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


# --- Change ownership: SHARED_DRIVE ---


def test_shared_drive_removed_change_matching_drive_id_passes() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-removed",
                        "time": "2024-01-03T12:00:00Z",
                        "driveId": _SHARED_DRIVE_ID,
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    page = reader.read_changes_page(
        scope=_SHARED_SCOPE,
        page_token=GoogleWorkspacePageToken(value="start"),
    )
    assert page.changes[0].removed is True
    assert page.changes[0].item is None


def test_shared_drive_removed_change_without_drive_id_fails() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-removed",
                        "time": "2024-01-03T12:00:00Z",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_SHARED_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_shared_drive_removed_change_wrong_drive_id_fails() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-removed",
                        "time": "2024-01-03T12:00:00Z",
                        "driveId": "other-drive",
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_SHARED_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_shared_drive_non_removed_change_without_item_drive_id_fails() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": False,
                        "fileId": "file-blob-1",
                        "time": "2024-01-03T12:00:00Z",
                        "driveId": _SHARED_DRIVE_ID,
                        "file": _blob_payload(),
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_SHARED_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


def test_shared_drive_change_item_drive_mismatch_fails() -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": False,
                        "fileId": "file-blob-1",
                        "time": "2024-01-03T12:00:00Z",
                        "driveId": _SHARED_DRIVE_ID,
                        "file": _blob_payload(drive_id="other-drive"),
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError):
        reader.read_changes_page(
            scope=_SHARED_SCOPE,
            page_token=GoogleWorkspacePageToken(value="start"),
        )


# --- Scope revalidation ---


def _invalid_constructed_scope() -> GoogleDriveScope:
    return GoogleDriveScope.model_construct(
        kind=GoogleDriveScopeKind.SHARED_DRIVE,
        drive_id=None,
    )


@pytest.mark.parametrize(
    "operation",
    [
        "read_items_page",
        "read_item",
        "read_start_page_token",
        "read_changes_page",
    ],
)
def test_reader_rejects_constructed_scope(operation: str) -> None:
    transport = _RecordingTransport(responses=[{"files": []}])
    reader = GoogleDriveKnowledgeReader(transport=transport)
    scope = _invalid_constructed_scope()
    with pytest.raises(IntegrationConfigurationError, match="invalid Google Drive scope"):
        if operation == "read_items_page":
            reader.read_items_page(scope=scope)
        elif operation == "read_item":
            reader.read_item(scope=scope, file_id="file-1")
        elif operation == "read_start_page_token":
            reader.read_start_page_token(scope=scope)
        else:
            reader.read_changes_page(
                scope=scope,
                page_token=GoogleWorkspacePageToken(value="start"),
            )
    assert transport.calls == []


def test_item_rejects_constructed_scope() -> None:
    invalid_scope = GoogleDriveScope.model_construct(
        kind=GoogleDriveScopeKind.USER,
        drive_id="unexpected-drive",
    )
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=invalid_scope,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=(),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
        )


def test_change_rejects_constructed_scope() -> None:
    invalid_scope = GoogleDriveScope.model_construct(
        kind="invalid",
        drive_id=None,
    )
    with pytest.raises(ValidationError):
        GoogleDriveChange(
            file_id="file-1",
            scope=invalid_scope,
            removed=True,
            changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )


def test_scope_canonical_copy_not_same_object() -> None:
    caller_scope = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
    item = GoogleDriveItem(
        remote_id="file-1",
        scope=caller_scope,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    assert item.scope == caller_scope
    assert item.scope is not caller_scope


# --- Canonical parent IDs ---


def test_parent_ids_tuple_trimmed() -> None:
    item = GoogleDriveItem(
        remote_id="file-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=("  parent-1  ", "parent-2"),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    assert item.parent_ids == ("parent-1", "parent-2")


def test_parent_ids_list_trimmed() -> None:
    parent_list = ["  parent-1  ", "parent-2"]
    item = GoogleDriveItem(
        remote_id="file-1",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=parent_list,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    assert item.parent_ids == ("parent-1", "parent-2")
    assert parent_list == ["  parent-1  ", "parent-2"]


def test_parent_ids_duplicate_after_trim_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDriveItem(
            remote_id="file-1",
            scope=_USER_SCOPE,
            kind=GoogleDriveItemKind.BLOB,
            name="doc.pdf",
            mime_type="application/pdf",
            parent_ids=("parent-1", " parent-1 "),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            version=1,
            can_download=True,
        )


# --- MIME and kind integrity ---


@pytest.mark.parametrize(
    ("kind", "mime_type"),
    [
        (GoogleDriveItemKind.FOLDER, "application/pdf"),
        (GoogleDriveItemKind.BLOB, "application/vnd.google-apps.document"),
        (GoogleDriveItemKind.NATIVE_DOCUMENT, "application/vnd.google-apps.folder"),
        (GoogleDriveItemKind.SHORTCUT, "application/pdf"),
        (GoogleDriveItemKind.OTHER, "application/octet-stream"),
    ],
)
def test_item_kind_mime_mismatch_rejected(
    kind: GoogleDriveItemKind,
    mime_type: str,
) -> None:
    kwargs: dict[str, object] = {
        "remote_id": "file-1",
        "scope": _USER_SCOPE,
        "kind": kind,
        "name": "doc",
        "mime_type": mime_type,
        "parent_ids": (),
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "modified_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
        "version": 1,
        "can_download": True,
    }
    if kind is GoogleDriveItemKind.SHORTCUT:
        kwargs["shortcut_target_id"] = "target-1"
        kwargs["shortcut_target_mime_type"] = "application/pdf"
    with pytest.raises(ValidationError):
        GoogleDriveItem(**kwargs)


# --- Removed changes ---


def test_removed_change_with_matching_item_valid() -> None:
    item = GoogleDriveItem(
        remote_id="file-removed",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    change = GoogleDriveChange(
        file_id="file-removed",
        scope=_USER_SCOPE,
        removed=True,
        changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        item=item,
    )
    assert change.item is item


def test_removed_change_item_remote_id_mismatch_rejected() -> None:
    item = GoogleDriveItem(
        remote_id="other-id",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    with pytest.raises(ValidationError):
        GoogleDriveChange(
            file_id="file-removed",
            scope=_USER_SCOPE,
            removed=True,
            changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
            item=item,
        )


def test_removed_change_item_scope_mismatch_rejected() -> None:
    item = GoogleDriveItem(
        remote_id="file-removed",
        scope=_USER_SCOPE,
        kind=GoogleDriveItemKind.BLOB,
        name="doc.pdf",
        mime_type="application/pdf",
        parent_ids=(),
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        modified_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        version=1,
        can_download=True,
    )
    with pytest.raises(ValidationError):
        GoogleDriveChange(
            file_id="file-removed",
            scope=_SHARED_SCOPE,
            removed=True,
            changed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
            item=item,
        )


# --- Strict RFC 3339 ---


@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-01-01T12:00:00Z",
        "2024-01-01T12:00:00.123Z",
        "2024-01-01T12:00:00+02:00",
        "2024-01-01T12:00:00.123456-05:30",
    ],
)
def test_rfc3339_valid_timestamps_accepted(timestamp: str) -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "drives": [
                    {
                        "id": "drive-1",
                        "name": "Team",
                        "createdTime": timestamp,
                        "hidden": False,
                    },
                ],
            },
            {"files": [_blob_payload()]},
            {
                "changes": [
                    {
                        "changeType": "file",
                        "removed": True,
                        "fileId": "file-1",
                        "time": timestamp,
                    },
                ],
                "newStartPageToken": "checkpoint",
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    drive_page = reader.list_shared_drives_page()
    assert drive_page.items[0].created_at.tzinfo is not None
    item_page = reader.read_items_page(scope=_USER_SCOPE)
    assert item_page.items[0].created_at.tzinfo is not None
    change_page = reader.read_changes_page(
        scope=_USER_SCOPE,
        page_token=GoogleWorkspacePageToken(value="start"),
    )
    assert change_page.changes[0].changed_at.tzinfo is not None


@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-01-01 12:00:00Z",
        "2024-01-01T12:00:00",
        "2024-01-01T12:00:00+0200",
        "2024-01-01T12:00:00z",
        " 2024-01-01T12:00:00Z",
        "2024-01-01T12:00:00Z ",
        "not-a-date",
    ],
)
def test_rfc3339_invalid_timestamps_rejected(timestamp: str) -> None:
    transport = _RecordingTransport(
        responses=[
            {
                "drives": [
                    {
                        "id": "drive-1",
                        "name": "Team",
                        "createdTime": timestamp,
                        "hidden": False,
                    },
                ],
            },
        ]
    )
    reader = GoogleDriveKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match="unexpected") as exc_info:
        reader.list_shared_drives_page()
    assert timestamp not in str(exc_info.value)
