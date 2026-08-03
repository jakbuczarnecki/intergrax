# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for GoogleWorkspaceDriveKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceBinaryTransport,
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
    GoogleDriveChange,
    GoogleDriveChangePage,
    GoogleDriveItem,
    GoogleDriveItemKind,
    GoogleDriveItemPage,
    GoogleDriveScope,
    GoogleDriveScopeKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive_content import (
    DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    GoogleDriveContentChanged,
    GoogleDriveContentMode,
    GoogleDriveContentTooLarge,
    GoogleDriveContentUnavailable,
    GoogleDriveFileContent,
    GoogleDriveUnsupportedContent,
    resolve_google_drive_content_profile,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
    GoogleWorkspacePageToken,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_DRIVE_CURSOR_VERSION,
    GOOGLE_DRIVE_ITEM_METADATA_VERSION,
    GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
    GOOGLE_DRIVE_USER_SCOPE_ID,
    GOOGLE_DRIVE_USER_SCOPE_TYPE,
    GoogleWorkspaceDriveKnowledgeAdapter,
    register_google_workspace_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    ConfluencePagesKnowledgeAdapter,
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_SHARED_DRIVE_ID = "shared-drive-abc"
_OTHER_SHARED_DRIVE_ID = "other-shared-drive"
_SECRET_START = "secret-start-token"
_SECRET_INVENTORY = "secret-inventory-page-token"
_SECRET_CHANGE = "secret-change-page-token"
_SECRET_NEW_START = "secret-new-start-token"

_USER_SCOPE = GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
_SHARED_SCOPE = GoogleDriveScope(
    kind=GoogleDriveScopeKind.SHARED_DRIVE,
    drive_id=_SHARED_DRIVE_ID,
)

_CREATED = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_MODIFIED = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
_TS = datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc)

_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "scope_kind",
        "drive_id",
        "item_kind",
        "mime_type",
        "parent_ids",
        "created_at",
        "modified_at",
        "size_bytes",
        "md5_checksum",
        "head_revision_id",
        "can_download",
        "shortcut_target_id",
        "shortcut_target_mime_type",
        "content_supported",
        "content_transport_mode",
        "content_mime_type",
    }
)


def _source(
    *,
    remote_scope_id: str = GOOGLE_DRIVE_USER_SCOPE_ID,
    remote_scope_type: str = GOOGLE_DRIVE_USER_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = GOOGLE_DRIVE_SOURCE_KIND,
    safe_display_name: str = "My Drive",
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name=safe_display_name,
            parameters=parameters or {},
        ),
    )


def _shared_drive_source() -> KnowledgeSourceRef:
    return _source(
        remote_scope_id=_SHARED_DRIVE_ID,
        remote_scope_type=GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
        safe_display_name="Team Drive",
    )


def _drive_item(
    *,
    remote_id: str = "file-1",
    scope: GoogleDriveScope = _USER_SCOPE,
    kind: GoogleDriveItemKind = GoogleDriveItemKind.BLOB,
    name: str = "report.pdf",
    mime_type: str = "application/pdf",
    parent_ids: tuple[str, ...] = ("parent-1",),
    drive_id: str | None = None,
    created_at: datetime = _CREATED,
    modified_at: datetime = _MODIFIED,
    size_bytes: int | None = 1024,
    md5_checksum: str | None = "checksum",
    version: int = 3,
    head_revision_id: str | None = "rev-1",
    web_view_link: str | None = "https://drive.google.com/file/d/file-1/view",
    can_download: bool = True,
    shortcut_target_id: str | None = None,
    shortcut_target_mime_type: str | None = None,
) -> GoogleDriveItem:
    if scope.kind is GoogleDriveScopeKind.SHARED_DRIVE:
        resolved_drive_id = drive_id or scope.drive_id
    else:
        resolved_drive_id = drive_id
    return GoogleDriveItem(
        remote_id=remote_id,
        scope=scope,
        kind=kind,
        name=name,
        mime_type=mime_type,
        parent_ids=parent_ids,
        drive_id=resolved_drive_id,
        created_at=created_at,
        modified_at=modified_at,
        size_bytes=size_bytes,
        md5_checksum=md5_checksum,
        version=version,
        head_revision_id=head_revision_id,
        web_view_link=web_view_link,
        can_download=can_download,
        shortcut_target_id=shortcut_target_id,
        shortcut_target_mime_type=shortcut_target_mime_type,
    )


def _item_page(
    *,
    items: tuple[GoogleDriveItem, ...],
    next_page_token: str | None = None,
) -> GoogleDriveItemPage:
    token = (
        GoogleWorkspacePageToken(value=next_page_token)
        if next_page_token is not None
        else None
    )
    return GoogleDriveItemPage(items=items, next_page_token=token)


def _drive_change(
    *,
    file_id: str,
    scope: GoogleDriveScope = _USER_SCOPE,
    removed: bool = False,
    item: GoogleDriveItem | None = None,
    changed_at: datetime = _MODIFIED,
) -> GoogleDriveChange:
    return GoogleDriveChange(
        file_id=file_id,
        scope=scope,
        removed=removed,
        changed_at=changed_at,
        item=item,
    )


def _change_page(
    *,
    changes: tuple[GoogleDriveChange, ...],
    next_page_token: str | None = None,
    new_start_page_token: str | None = None,
) -> GoogleDriveChangePage:
    next_token = (
        GoogleWorkspacePageToken(value=next_page_token)
        if next_page_token is not None
        else None
    )
    new_start = (
        GoogleWorkspacePageToken(value=new_start_page_token)
        if new_start_page_token is not None
        else None
    )
    return GoogleDriveChangePage(
        changes=changes,
        next_page_token=next_token,
        new_start_page_token=new_start,
    )


def _file_content(
    *,
    item: GoogleDriveItem,
    data: bytes = b"hello",
    mode: GoogleDriveContentMode | None = None,
    content_mime_type: str | None = None,
) -> GoogleDriveFileContent:
    profile = resolve_google_drive_content_profile(item)
    resolved_mode = mode or profile.mode
    resolved_mime = content_mime_type or profile.content_mime_type
    return GoogleDriveFileContent(
        item=item,
        mode=resolved_mode,
        content_mime_type=resolved_mime,
        data=data,
        size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
    )


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return KnowledgeCursor(value=encoded, version=GOOGLE_DRIVE_CURSOR_VERSION)


def _changes_cursor(
    *,
    scope_kind: str = "user",
    drive_id: str | None = None,
    change_page_token: str = _SECRET_CHANGE,
) -> KnowledgeCursor:
    return _encode_cursor(
        {
            "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
            "scope_kind": scope_kind,
            "drive_id": drive_id,
            "phase": "changes",
            "inventory_page_token": None,
            "change_page_token": change_page_token,
        }
    )


def _inventory_cursor(
    *,
    scope_kind: str = "user",
    drive_id: str | None = None,
    inventory_page_token: str = _SECRET_INVENTORY,
    change_page_token: str = _SECRET_START,
) -> KnowledgeCursor:
    return _encode_cursor(
        {
            "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
            "scope_kind": scope_kind,
            "drive_id": drive_id,
            "phase": "inventory",
            "inventory_page_token": inventory_page_token,
            "change_page_token": change_page_token,
        }
    )


def _item_metadata(item: GoogleDriveItem) -> dict[str, Any]:
    try:
        profile = resolve_google_drive_content_profile(item)
        content_supported = True
        content_transport_mode = profile.mode.value
        content_mime_type = profile.content_mime_type
    except GoogleDriveUnsupportedContent:
        content_supported = False
        content_transport_mode = None
        content_mime_type = None
    return {
        "schema_version": GOOGLE_DRIVE_ITEM_METADATA_VERSION,
        "scope_kind": item.scope.kind.value,
        "drive_id": item.drive_id,
        "item_kind": item.kind.value,
        "mime_type": item.mime_type,
        "parent_ids": list(item.parent_ids),
        "created_at": item.created_at.isoformat(),
        "modified_at": item.modified_at.isoformat(),
        "size_bytes": item.size_bytes,
        "md5_checksum": item.md5_checksum,
        "head_revision_id": item.head_revision_id,
        "can_download": item.can_download,
        "shortcut_target_id": item.shortcut_target_id,
        "shortcut_target_mime_type": item.shortcut_target_mime_type,
        "content_supported": content_supported,
        "content_transport_mode": content_transport_mode,
        "content_mime_type": content_mime_type,
    }


def _descriptor_for_item(item: GoogleDriveItem) -> KnowledgeItemDescriptor:
    metadata = _item_metadata(item)
    content_supported = metadata["content_supported"]
    if content_supported:
        content_mode = KnowledgeContentMode.BINARY
        content_available = item.can_download
    else:
        content_mode = KnowledgeContentMode.STRUCTURED_RECORD
        content_available = False
    parent_remote_id = item.parent_ids[0] if len(item.parent_ids) == 1 else None
    item_type_map = {
        GoogleDriveItemKind.BLOB: "google_workspace_drive_blob",
        GoogleDriveItemKind.FOLDER: "google_workspace_drive_folder",
        GoogleDriveItemKind.NATIVE_DOCUMENT: "google_workspace_drive_native_document",
        GoogleDriveItemKind.SHORTCUT: "google_workspace_drive_shortcut",
        GoogleDriveItemKind.OTHER: "google_workspace_drive_other",
    }
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(
            remote_id=item.remote_id,
            parent_remote_id=parent_remote_id,
            logical_key=None,
        ),
        revision=KnowledgeItemRevision(
            version=str(item.version),
            etag=None,
            content_hash=None,
            acl_hash=None,
            updated_at=item.modified_at,
        ),
        title=item.name,
        item_type=item_type_map[item.kind],
        content_mode=content_mode,
        content_available=content_available,
        provenance=KnowledgeItemProvenance(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            remote_id=item.remote_id,
            web_url=item.web_view_link,
            safe_locator=None,
        ),
        metadata=metadata,
    )


class _FakeGoogleWorkspaceIntegration(CollaborationSuite):
    def __init__(
        self,
        *,
        start_token: GoogleWorkspacePageToken | None = None,
        inventory_pages: list[GoogleDriveItemPage] | None = None,
        change_pages: list[GoogleDriveChangePage] | None = None,
        content_by_id: dict[str, GoogleDriveFileContent] | None = None,
    ) -> None:
        self._start_token = start_token or GoogleWorkspacePageToken(value=_SECRET_START)
        self._inventory_pages = list(inventory_pages or [])
        self._change_pages = list(change_pages or [])
        self._content_by_id = dict(content_by_id or {})
        self.start_token_calls: list[dict[str, Any]] = []
        self.inventory_calls: list[dict[str, Any]] = []
        self.change_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []

    def read_drive_start_page_token(
        self,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleWorkspacePageToken:
        self.start_token_calls.append({"scope": scope})
        return self._start_token

    def read_drive_items_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken | None,
        limit: int,
    ) -> GoogleDriveItemPage:
        self.inventory_calls.append(
            {"scope": scope, "page_token": page_token, "limit": limit}
        )
        if not self._inventory_pages:
            return GoogleDriveItemPage(items=())
        return self._inventory_pages.pop(0)

    def read_drive_changes_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken,
        limit: int,
    ) -> GoogleDriveChangePage:
        self.change_calls.append(
            {"scope": scope, "page_token": page_token, "limit": limit}
        )
        if not self._change_pages:
            raise IntegrationDependencyError("no change pages configured")
        return self._change_pages.pop(0)

    def read_drive_file_content(
        self,
        *,
        item: GoogleDriveItem,
        max_bytes: int,
    ) -> GoogleDriveFileContent:
        self.content_calls.append({"item": item, "max_bytes": max_bytes})
        return self._content_by_id[item.remote_id]

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


class _StubBinaryTransport(GoogleWorkspaceBinaryTransport):
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError("drive reads are bound to the injected fake client")

    def get_binary(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        raise NotImplementedError("drive content reads are bound to the injected fake client")


class _StubClientFamily:
    def __init__(self) -> None:
        self.transport: GoogleWorkspaceTransport = _StubBinaryTransport()


class _BoundGoogleWorkspaceIntegration(GoogleWorkspaceCollaborationSuiteIntegration):
    _bound_fake: _FakeGoogleWorkspaceIntegration = PrivateAttr()

    @classmethod
    def from_fake(cls, fake: _FakeGoogleWorkspaceIntegration) -> _BoundGoogleWorkspaceIntegration:
        bound = cls.from_client(_StubClientFamily(), enabled=True)
        bound._bound_fake = fake
        return bound

    def read_drive_start_page_token(
        self,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleWorkspacePageToken:
        return self._bound_fake.read_drive_start_page_token(scope=scope)

    def read_drive_items_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 200,
    ) -> GoogleDriveItemPage:
        return self._bound_fake.read_drive_items_page(
            scope=scope,
            page_token=page_token,
            limit=limit,
        )

    def read_drive_changes_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken,
        limit: int = 200,
    ) -> GoogleDriveChangePage:
        return self._bound_fake.read_drive_changes_page(
            scope=scope,
            page_token=page_token,
            limit=limit,
        )

    def read_drive_file_content(
        self,
        *,
        item: GoogleDriveItem,
        max_bytes: int = DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    ) -> GoogleDriveFileContent:
        return self._bound_fake.read_drive_file_content(
            item=item,
            max_bytes=max_bytes,
        )


def _integration(fake: _FakeGoogleWorkspaceIntegration) -> GoogleWorkspaceCollaborationSuiteIntegration:
    return _BoundGoogleWorkspaceIntegration.from_fake(fake)


def _assert_no_secrets_in_rendered(rendered: str) -> None:
    for secret in (
        _SECRET_START,
        _SECRET_INVENTORY,
        _SECRET_CHANGE,
        _SECRET_NEW_START,
    ):
        assert secret not in rendered


def _assert_invalid_descriptor_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    assert err.__cause__ is None
    _assert_no_secrets_in_rendered(f"{err!r} {err.safe_message}")
    for secret in (
        "file-1",
        "report.pdf",
        "drive.google.com",
        "2024-01-01",
        "checksum",
    ):
        assert secret not in f"{err!r} {err.safe_message}"


def _assert_canonical_error_identity(err: VendorKnowledgeError) -> None:
    assert err.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert err.source_kind == GOOGLE_DRIVE_SOURCE_KIND


def _assert_invalid_cursor_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    fake: _FakeGoogleWorkspaceIntegration,
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert err.retryable is False
    _assert_canonical_error_identity(err)
    assert fake.start_token_calls == []
    assert fake.inventory_calls == []
    assert fake.change_calls == []
    _assert_no_secrets_in_rendered(f"{err!r} {err.safe_message}")


def _assert_invalid_scope_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    fake: _FakeGoogleWorkspaceIntegration,
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    _assert_canonical_error_identity(err)
    assert fake.start_token_calls == []
    assert fake.inventory_calls == []
    assert fake.change_calls == []
    assert fake.content_calls == []
    _assert_no_secrets_in_rendered(f"{err!r} {err.safe_message}")


def _assert_invalid_provider_response_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    fake: _FakeGoogleWorkspaceIntegration,
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert err.retryable is False
    _assert_canonical_error_identity(err)
    _assert_no_secrets_in_rendered(f"{err!r} {err.safe_message}")


async def _fetch_content_invalid_descriptor(
    item: KnowledgeItemDescriptor,
) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={
            "file-1": _file_content(item=_drive_item(), data=b"x"),
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    assert fake.content_calls == []
    _assert_invalid_descriptor_boundary(exc_info)
    _assert_canonical_error_identity(exc_info.value)
    return exc_info


async def test_adapter_identity() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    assert adapter.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == GOOGLE_DRIVE_SOURCE_KIND


async def test_capabilities_exact_set() -> None:
    caps = GoogleWorkspaceDriveKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is True
    assert caps.content_fetch is True
    assert caps.binary_content is True
    assert caps.rich_text_content is False
    assert caps.structured_content is False
    assert caps.permissions is False
    assert caps.tombstones is True
    assert caps.remote_versions is True
    assert caps.reconciliation is True


async def test_valid_user_scope_inspect() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    info = await adapter.inspect_scope(integration=_integration(fake), source=_source())
    assert info.safe_display_name == "My Drive"
    assert fake.start_token_calls == []
    assert fake.inventory_calls == []
    assert fake.change_calls == []


async def test_valid_shared_drive_scope_inspect() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    info = await adapter.inspect_scope(
        integration=_integration(fake),
        source=_shared_drive_source(),
    )
    assert info.safe_display_name == "Team Drive"
    assert fake.start_token_calls == []
    assert fake.inventory_calls == []
    assert fake.change_calls == []


@pytest.mark.parametrize(
    ("source",),
    [
        (_source(provider_id="jira"),),
        (_source(integration_kind=IntegrationCategory.ISSUE_TRACKER),),
        (_source(source_kind="mail"),),
        (_source(remote_scope_type="sharepoint"),),
        (_source(parameters={"site": "x"}),),
        (_source(remote_scope_id="wrong-user"),),
        (
            _source(
                remote_scope_id=_SHARED_DRIVE_ID,
                remote_scope_type=GOOGLE_DRIVE_USER_SCOPE_TYPE,
            ),
        ),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=source,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_malformed_shared_drive_id_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=_source(
                remote_scope_id="\x00bad",
                remote_scope_type=GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
            ),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_empty_shared_drive_id_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DRIVE_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="",
            remote_scope_type=GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
            safe_display_name="Team Drive",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=source,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_whitespace_shared_drive_id_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DRIVE_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="   ",
            remote_scope_type=GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
            safe_display_name="Team Drive",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=source,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_wrong_integration_object_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=object(), source=_source())
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_initial_inventory_calls_start_token_first() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[
            _item_page(items=(_drive_item(),)),
        ]
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert len(fake.start_token_calls) == 1
    assert len(fake.inventory_calls) == 1
    assert fake.start_token_calls[0]["scope"] == _USER_SCOPE
    assert fake.inventory_calls[0]["page_token"] is None


async def test_initial_inventory_user_scope() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="user-file-1")
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.changes[0].remote_id == "user-file-1"
    assert fake.inventory_calls[0]["scope"] == _USER_SCOPE


async def test_initial_inventory_shared_drive_scope() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="shared-file-1", scope=_SHARED_SCOPE)
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_shared_drive_source(),
        cursor=None,
        limit=50,
    )
    assert page.changes[0].remote_id == "shared-file-1"
    assert fake.inventory_calls[0]["scope"] == _SHARED_SCOPE


async def test_initial_inventory_continuation_cursor() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[
            _item_page(items=(_drive_item(),), next_page_token=_SECRET_INVENTORY),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.has_more is True
    assert page.next_cursor is not None
    assert page.proposed_checkpoint == page.next_cursor
    _assert_no_secrets_in_rendered(page.next_cursor.value)
    _assert_no_secrets_in_rendered(repr(page.next_cursor))


async def test_initial_inventory_final_checkpoint_to_changes() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(_drive_item(),))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    _assert_no_secrets_in_rendered(page.proposed_checkpoint.value)


async def test_initial_inventory_empty() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=())],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.changes == ()
    assert page.has_more is False
    assert page.proposed_checkpoint is not None


@pytest.mark.parametrize("limit", [1, 200, 201, 1000])
async def test_limit_accepted_and_clamped(limit: int) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(_drive_item(),))],
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=limit,
    )
    assert fake.inventory_calls[0]["limit"] == min(limit, 200)


@pytest.mark.parametrize("limit", [0, 1001, True, False, "10", 10.5, None])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=_source(),
            cursor=None,
            limit=limit,  # type: ignore[arg-type]
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


async def test_inventory_continuation_reads_next_page() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[
            _item_page(items=(_drive_item(remote_id="page-2-item"),)),
        ]
    )
    cursor = _inventory_cursor()
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    assert fake.start_token_calls == []
    assert len(fake.inventory_calls) == 1
    assert fake.inventory_calls[0]["page_token"].value == _SECRET_INVENTORY


async def test_changes_upsert_mapping() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="changed-1")
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(
                changes=(
                    _drive_change(
                        file_id="changed-1",
                        item=item,
                    ),
                ),
                new_start_page_token=_SECRET_NEW_START,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(),
        limit=50,
    )
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.UPSERT
    assert change.remote_id == "changed-1"
    assert change.descriptor is not None
    assert change.descriptor.identity.remote_id == "changed-1"


async def test_changes_deleted_tombstone() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(
                changes=(
                    _drive_change(file_id="gone-1", removed=True),
                ),
                new_start_page_token=_SECRET_NEW_START,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(),
        limit=50,
    )
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.DELETED
    assert change.remote_id == "gone-1"
    assert change.descriptor is None


async def test_changes_ordering_preserved() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    items = tuple(
        _drive_item(remote_id=f"ordered-{index}")
        for index in range(3)
    )
    changes = tuple(
        _drive_change(file_id=item.remote_id, item=item)
        for item in items
    )
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(changes=changes, new_start_page_token=_SECRET_NEW_START),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(),
        limit=50,
    )
    assert [change.remote_id for change in page.changes] == [
        "ordered-0",
        "ordered-1",
        "ordered-2",
    ]


async def test_changes_no_dedup() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="dup-change")
    changes = (
        _drive_change(file_id="dup-change", item=item),
        _drive_change(file_id="dup-change", item=item),
    )
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(changes=changes, new_start_page_token=_SECRET_NEW_START),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(),
        limit=50,
    )
    assert len(page.changes) == 2
    assert page.changes[0].remote_id == "dup-change"
    assert page.changes[1].remote_id == "dup-change"


async def test_changes_next_page_cursor() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(
                changes=(
                    _drive_change(
                        file_id="changed-1",
                        item=_drive_item(remote_id="changed-1"),
                    ),
                ),
                next_page_token=_SECRET_CHANGE,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(change_page_token=_SECRET_START),
        limit=50,
    )
    assert page.has_more is True
    assert page.next_cursor is not None
    assert page.proposed_checkpoint == page.next_cursor
    _assert_no_secrets_in_rendered(page.next_cursor.value)


async def test_changes_new_start_checkpoint() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(changes=(), new_start_page_token=_SECRET_NEW_START),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(),
        limit=50,
    )
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    _assert_no_secrets_in_rendered(page.proposed_checkpoint.value)


async def test_changes_phase_no_inventory_or_start_calls() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            _change_page(changes=(), new_start_page_token=_SECRET_NEW_START),
        ]
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=_changes_cursor(),
        limit=50,
    )
    assert fake.start_token_calls == []
    assert fake.inventory_calls == []
    assert len(fake.change_calls) == 1


async def test_changes_malformed_page_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="mismatch-id")
    malformed_change = GoogleDriveChange.model_construct(
        file_id="different-id",
        scope=_USER_SCOPE,
        removed=False,
        changed_at=_MODIFIED,
        item=item,
    )
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[
            GoogleDriveChangePage.model_construct(
                changes=(malformed_change,),
                next_page_token=None,
                new_start_page_token=GoogleWorkspacePageToken(value=_SECRET_NEW_START),
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_changes_missing_new_start_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        change_pages=[GoogleDriveChangePage.model_construct(changes=(), next_page_token=None, new_start_page_token=None)],
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.parametrize(
    ("cursor",),
    [
        (KnowledgeCursor(value="not-base64", version=GOOGLE_DRIVE_CURSOR_VERSION),),
        (KnowledgeCursor(value="!!!", version=GOOGLE_DRIVE_CURSOR_VERSION),),
        (KnowledgeCursor(value=_encode_cursor({"bad": 1}).value, version="other.v1"),),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
                        "scope_kind": "user",
                        "drive_id": "should-not-be-here",
                        "phase": "inventory",
                        "inventory_page_token": _SECRET_INVENTORY,
                        "change_page_token": _SECRET_START,
                    }
                ).value,
                version=GOOGLE_DRIVE_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
                        "scope_kind": "shared_drive",
                        "drive_id": None,
                        "phase": "changes",
                        "inventory_page_token": None,
                        "change_page_token": _SECRET_CHANGE,
                    }
                ).value,
                version=GOOGLE_DRIVE_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
                        "scope_kind": "user",
                        "drive_id": None,
                        "phase": "inventory",
                        "inventory_page_token": None,
                        "change_page_token": _SECRET_START,
                    }
                ).value,
                version=GOOGLE_DRIVE_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
                        "scope_kind": "user",
                        "drive_id": None,
                        "phase": "changes",
                        "inventory_page_token": _SECRET_INVENTORY,
                        "change_page_token": _SECRET_CHANGE,
                    }
                ).value,
                version=GOOGLE_DRIVE_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
                        "scope_kind": "shared_drive",
                        "drive_id": _OTHER_SHARED_DRIVE_ID,
                        "phase": "changes",
                        "inventory_page_token": None,
                        "change_page_token": _SECRET_CHANGE,
                    }
                ).value,
                version=GOOGLE_DRIVE_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
                        "scope_kind": "user",
                        "drive_id": None,
                        "phase": "changes",
                        "inventory_page_token": None,
                        "change_page_token": "",
                    }
                ).value,
                version=GOOGLE_DRIVE_CURSOR_VERSION,
            ),
        ),
    ],
)
async def test_invalid_cursor_rejected(cursor: KnowledgeCursor) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    _assert_no_secrets_in_rendered(str(exc_info.value))
    _assert_no_secrets_in_rendered(repr(exc_info.value))


async def test_invalid_shared_drive_cursor_for_user_source() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    cursor = _encode_cursor(
        {
            "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
            "scope_kind": "shared_drive",
            "drive_id": _SHARED_DRIVE_ID,
            "phase": "changes",
            "inventory_page_token": None,
            "change_page_token": _SECRET_CHANGE,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeGoogleWorkspaceIntegration()),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_blob_descriptor_mapping() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="blob-1", parent_ids=("parent-1",))
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.remote_id == "blob-1"
    assert descriptor.identity.parent_remote_id == "parent-1"
    assert descriptor.title == "report.pdf"
    assert descriptor.item_type == "google_workspace_drive_blob"
    assert descriptor.content_mode is KnowledgeContentMode.BINARY
    assert descriptor.content_available is True
    assert descriptor.revision.version == "3"
    assert descriptor.revision.updated_at == _MODIFIED
    assert descriptor.provenance.web_url == "https://drive.google.com/file/d/file-1/view"


async def test_native_descriptor_mapping() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(
        remote_id="gdoc-1",
        kind=GoogleDriveItemKind.NATIVE_DOCUMENT,
        name="Notes",
        mime_type="application/vnd.google-apps.document",
        size_bytes=0,
        md5_checksum=None,
    )
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == "google_workspace_drive_native_document"
    assert descriptor.content_mode is KnowledgeContentMode.BINARY
    assert descriptor.content_available is True
    assert descriptor.metadata is not None
    assert descriptor.metadata["content_transport_mode"] == "export"
    assert descriptor.metadata["content_mime_type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


async def test_folder_descriptor_mapping() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(
        remote_id="folder-1",
        kind=GoogleDriveItemKind.FOLDER,
        name="Folder",
        mime_type="application/vnd.google-apps.folder",
        parent_ids=(),
        size_bytes=None,
        md5_checksum=None,
        head_revision_id=None,
        web_view_link=None,
        can_download=False,
        version=1,
    )
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == "google_workspace_drive_folder"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is False
    assert descriptor.metadata is not None
    assert descriptor.metadata["content_supported"] is False


async def test_shortcut_descriptor_mapping() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(
        remote_id="shortcut-1",
        kind=GoogleDriveItemKind.SHORTCUT,
        name="Shortcut",
        mime_type="application/vnd.google-apps.shortcut",
        size_bytes=None,
        md5_checksum=None,
        shortcut_target_id="target-1",
        shortcut_target_mime_type="application/pdf",
    )
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == "google_workspace_drive_shortcut"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is False
    assert descriptor.metadata is not None
    assert descriptor.metadata["shortcut_target_id"] == "target-1"


async def test_unsupported_native_descriptor_mapping() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(
        remote_id="form-1",
        kind=GoogleDriveItemKind.NATIVE_DOCUMENT,
        name="Form",
        mime_type="application/vnd.google-apps.form",
        size_bytes=0,
        md5_checksum=None,
        can_download=False,
    )
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == "google_workspace_drive_native_document"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is False
    assert descriptor.metadata is not None
    assert descriptor.metadata["content_supported"] is False


async def test_parent_projection_single_parent() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="child-1", parent_ids=("only-parent",))
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.parent_remote_id == "only-parent"


async def test_parent_projection_multiple_parents_null() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="child-2", parent_ids=("parent-a", "parent-b"))
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.parent_remote_id is None


async def test_metadata_schema() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.metadata is not None
    assert set(descriptor.metadata.keys()) == _METADATA_KEYS
    assert descriptor.metadata["schema_version"] == GOOGLE_DRIVE_ITEM_METADATA_VERSION


async def test_cross_scope_inventory_item_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(scope=_SHARED_SCOPE, remote_id="wrong-scope")
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[_item_page(items=(item,))],
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_duplicate_inventory_remote_id_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="dup-1")
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[
            GoogleDriveItemPage.model_construct(
                items=(item, item),
                next_page_token=None,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_fetch_content_valid_blob() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="file-1")
    data = b"\x00\xffbinary"
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={"file-1": _file_content(item=item, data=data)},
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_descriptor_for_item(item),
    )
    assert content.mode is KnowledgeContentMode.BINARY
    assert content.binary == data
    assert content.mime_type == "application/pdf"
    assert content.content_hash == hashlib.sha256(data).hexdigest()
    assert fake.content_calls[0]["item"].remote_id == "file-1"
    assert fake.content_calls[0]["max_bytes"] == DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES


async def test_fetch_content_valid_export() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(
        remote_id="gdoc-1",
        kind=GoogleDriveItemKind.NATIVE_DOCUMENT,
        name="Notes",
        mime_type="application/vnd.google-apps.document",
        size_bytes=0,
        md5_checksum=None,
    )
    data = b"exported-docx"
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={"gdoc-1": _file_content(item=item, data=data)},
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_descriptor_for_item(item),
    )
    assert content.mode is KnowledgeContentMode.BINARY
    assert content.binary == data
    assert content.mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


async def test_fetch_content_provider_mismatch_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="file-1")
    wrong_item = _drive_item(remote_id="other-file")
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={"file-1": _file_content(item=wrong_item, data=b"x")},
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_descriptor_for_item(item),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_fetch_content_rejects_non_downloadable() -> None:
    item = _drive_item(remote_id="file-1", can_download=False)
    await _fetch_content_invalid_descriptor(_descriptor_for_item(item))


async def test_fetch_content_rejects_folder() -> None:
    item = _drive_item(
        remote_id="folder-1",
        kind=GoogleDriveItemKind.FOLDER,
        name="Folder",
        mime_type="application/vnd.google-apps.folder",
        parent_ids=(),
        size_bytes=None,
        md5_checksum=None,
        head_revision_id=None,
        web_view_link=None,
        can_download=False,
        version=1,
    )
    await _fetch_content_invalid_descriptor(_descriptor_for_item(item))


async def test_fetch_content_rejects_wrong_provenance() -> None:
    descriptor = _descriptor_for_item(_drive_item())
    bad_provenance = KnowledgeItemProvenance(
        provider_id="jira",
        source_kind="issues",
        remote_id="file-1",
    )
    bad = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=bad_provenance,
        metadata=descriptor.metadata,
    )
    await _fetch_content_invalid_descriptor(bad)


@pytest.mark.parametrize(
    "created_at",
    [
        123,
        [],
        {},
        "",
        "   ",
        "not-a-date",
        "2024-01-01T12:00:00",
    ],
)
async def test_fetch_content_rejects_malformed_created_at(created_at: object) -> None:
    descriptor = _descriptor_for_item(_drive_item())
    metadata = dict(descriptor.metadata or {})
    metadata["created_at"] = created_at
    bad = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=metadata,
    )
    await _fetch_content_invalid_descriptor(bad)


async def test_fetch_content_accepts_created_at_with_timezone_offset() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item()
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={"file-1": _file_content(item=item, data=b"x")},
    )
    descriptor = _descriptor_for_item(item)
    metadata = dict(descriptor.metadata or {})
    metadata["created_at"] = "2024-01-01T14:00:00+02:00"
    item_desc = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=metadata,
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=item_desc,
    )
    provider_item = fake.content_calls[0]["item"]
    assert provider_item.created_at.tzinfo is not None
    assert provider_item.created_at.utcoffset() is not None


async def test_fetch_content_accepts_modified_at_with_z_suffix() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    modified = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    item = _drive_item(modified_at=modified)
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={"file-1": _file_content(item=item, data=b"x")},
    )
    descriptor = _descriptor_for_item(item)
    metadata = dict(descriptor.metadata or {})
    metadata["modified_at"] = "2024-01-02T12:00:00Z"
    item_desc = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=KnowledgeItemRevision(
            version=descriptor.revision.version,
            etag=None,
            content_hash=None,
            acl_hash=None,
            updated_at=modified,
        ),
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=metadata,
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=item_desc,
    )
    assert fake.content_calls[0]["item"].modified_at == modified


async def test_fetch_content_malformed_result_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item(remote_id="file-1")
    bad_content = GoogleDriveFileContent.model_construct(
        item=item,
        mode=GoogleDriveContentMode.BLOB,
        content_mime_type="application/pdf",
        data=b"x",
        size_bytes=99,
        content_hash=hashlib.sha256(b"x").hexdigest(),
    )
    fake = _FakeGoogleWorkspaceIntegration(content_by_id={"file-1": bad_content})
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_descriptor_for_item(item),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_fetch_permissions_unsupported() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_descriptor_for_item(_drive_item()),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert fake.content_calls == []


async def test_registry_registration() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = register_google_workspace_drive_knowledge_adapter(registry)
    assert isinstance(adapter, GoogleWorkspaceDriveKnowledgeAdapter)
    resolved = registry.resolve(source=_source())
    assert resolved is adapter
    jira_registry = KnowledgeAdapterRegistry()
    register_jira_issues_knowledge_adapter(jira_registry)
    jira_resolved = jira_registry.resolve(
        source=KnowledgeSourceRef(
            tenant_id="tenant-1",
            provider_id="jira",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            scope=KnowledgeSourceScope(
                remote_scope_id="PROJ",
                remote_scope_type="jira_project",
                safe_display_name="Project",
                parameters={},
            ),
        )
    )
    assert isinstance(jira_resolved, JiraIssuesKnowledgeAdapter)
    confluence_registry = KnowledgeAdapterRegistry()
    register_confluence_pages_knowledge_adapter(confluence_registry)
    confluence_resolved = confluence_registry.resolve(
        source=KnowledgeSourceRef(
            tenant_id="tenant-1",
            provider_id="confluence",
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
            source_kind="pages",
            scope=KnowledgeSourceScope(
                remote_scope_id="10000",
                remote_scope_type="confluence_space",
                safe_display_name="Space",
                parameters={},
            ),
        )
    )
    assert isinstance(confluence_resolved, ConfluencePagesKnowledgeAdapter)


async def test_duplicate_registry_registration_rejected() -> None:
    registry = KnowledgeAdapterRegistry()
    register_google_workspace_drive_knowledge_adapter(registry)
    with pytest.raises(ValueError):
        register_google_workspace_drive_knowledge_adapter(registry)


async def test_package_export() -> None:
    from intergrax.runtime.vendor_knowledge.adapters import (
        GOOGLE_DRIVE_CURSOR_VERSION as exported_cursor_version,
        GOOGLE_DRIVE_ITEM_METADATA_VERSION as exported_metadata_version,
        GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE as exported_shared_scope_type,
        GOOGLE_DRIVE_USER_SCOPE_ID as exported_user_scope_id,
        GOOGLE_DRIVE_USER_SCOPE_TYPE as exported_user_scope_type,
        GoogleWorkspaceDriveKnowledgeAdapter as exported_adapter,
        register_google_workspace_drive_knowledge_adapter as exported_register,
    )

    assert exported_cursor_version == GOOGLE_DRIVE_CURSOR_VERSION
    assert exported_metadata_version == GOOGLE_DRIVE_ITEM_METADATA_VERSION
    assert exported_shared_scope_type == GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE
    assert exported_user_scope_id == GOOGLE_DRIVE_USER_SCOPE_ID
    assert exported_user_scope_type == GOOGLE_DRIVE_USER_SCOPE_TYPE
    assert exported_adapter is GoogleWorkspaceDriveKnowledgeAdapter
    registry = KnowledgeAdapterRegistry()
    registered = exported_register(registry)
    assert isinstance(registered, GoogleWorkspaceDriveKnowledgeAdapter)


async def test_integration_dependency_error_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_start_page_token(self, *, scope):
            raise IntegrationDependencyError("boom")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "boom" not in str(exc_info.value)


async def test_integration_configuration_error_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_start_page_token(self, *, scope):
            raise IntegrationConfigurationError("misconfigured")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    ("kind", "expected_code", "retryable"),
    [
        (GoogleWorkspaceErrorKind.AUTHENTICATION, VendorKnowledgeErrorCode.AUTHENTICATION_FAILED, False),
        (GoogleWorkspaceErrorKind.AUTHORIZATION, VendorKnowledgeErrorCode.AUTHORIZATION_DENIED, False),
        (GoogleWorkspaceErrorKind.NOT_FOUND, VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND, False),
        (GoogleWorkspaceErrorKind.RATE_LIMITED, VendorKnowledgeErrorCode.RATE_LIMITED, True),
        (GoogleWorkspaceErrorKind.TEMPORARY, VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE, True),
        (GoogleWorkspaceErrorKind.MALFORMED_RESPONSE, VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE, False),
        (GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT, VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE, False),
        (GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE, VendorKnowledgeErrorCode.CONFIGURATION_ERROR, False),
    ],
)
async def test_google_api_error_mapping(
    kind: GoogleWorkspaceErrorKind,
    expected_code: VendorKnowledgeErrorCode,
    retryable: bool,
) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_start_page_token(self, *, scope):
            raise GoogleWorkspaceApiError(
                kind=kind,
                status_code=500,
                retry_after_seconds=None,
                safe_reason="secret-provider-detail",
                attempts=1,
            )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is expected_code
    assert exc_info.value.retryable is retryable
    assert "secret-provider-detail" not in str(exc_info.value)
    assert "secret-provider-detail" not in repr(exc_info.value)


async def test_google_invalid_request_configuration_error() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_start_page_token(self, *, scope):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=400,
                retry_after_seconds=None,
                safe_reason="bad-request",
                attempts=1,
            )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


async def test_google_invalid_request_cursor_error() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_changes_page(self, *, scope, page_token, limit):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=410,
                retry_after_seconds=None,
                safe_reason="stale-token",
                attempts=1,
            )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    _assert_no_secrets_in_rendered(str(exc_info.value))


async def test_content_changed_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_file_content(self, *, item, max_bytes):
            raise GoogleDriveContentChanged()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_FailingSuite()),
            source=_source(),
            item=_descriptor_for_item(item),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True


async def test_content_too_large_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_file_content(self, *, item, max_bytes):
            raise GoogleDriveContentTooLarge()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_FailingSuite()),
            source=_source(),
            item=_descriptor_for_item(item),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False


async def test_content_unavailable_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_file_content(self, *, item, max_bytes):
            raise GoogleDriveContentUnavailable()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_FailingSuite()),
            source=_source(),
            item=_descriptor_for_item(item),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.AUTHORIZATION_DENIED
    assert exc_info.value.retryable is False


async def test_unsupported_content_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    item = _drive_item()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_file_content(self, *, item, max_bytes):
            raise GoogleDriveUnsupportedContent()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_FailingSuite()),
            source=_source(),
            item=_descriptor_for_item(item),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False


async def test_generic_exception_translated() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_drive_start_page_token(self, *, scope):
            raise RuntimeError("unexpected runtime failure")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "unexpected runtime failure" not in str(exc_info.value)


async def test_public_objects_do_not_leak_secrets() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration(
        inventory_pages=[
            _item_page(items=(_drive_item(),), next_page_token=_SECRET_INVENTORY),
        ],
        content_by_id={
            "file-1": _file_content(item=_drive_item(), data=b"hello"),
        },
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    blob = json.dumps(page.model_dump(mode="json"))
    _assert_no_secrets_in_rendered(blob)
    item = _drive_item()
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_descriptor_for_item(item),
    )
    content_blob = json.dumps(content.model_dump(mode="json"))
    _assert_no_secrets_in_rendered(content_blob)
    err = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_CURSOR,
        safe_message="Google Workspace Drive knowledge cursor is invalid",
        provider_id=adapter.provider_id,
        source_kind=adapter.source_kind,
        retryable=False,
    )
    _assert_no_secrets_in_rendered(repr(err))
    _assert_no_secrets_in_rendered(str(err))


class _KnowledgeSourceRefSubclass(KnowledgeSourceRef):
    pass


class _KnowledgeCursorSubclass(KnowledgeCursor):
    pass


class _KnowledgeItemDescriptorSubclass(KnowledgeItemDescriptor):
    pass


class _GoogleDriveItemPageSubclass(GoogleDriveItemPage):
    pass


class _GoogleDriveChangePageSubclass(GoogleDriveChangePage):
    pass


class _GoogleWorkspacePageTokenSubclass(GoogleWorkspacePageToken):
    pass


class _HostileStr(str):
  def __str__(self) -> str:
    raise RuntimeError("hostile str")


class _HostileDatetime(datetime):
  def isoformat(self) -> str:
    raise RuntimeError("hostile datetime")


def _valid_changes_cursor_value() -> str:
    return _changes_cursor().value


@pytest.mark.parametrize(
    "cursor",
    [
        KnowledgeCursor(value="!!!!" + _valid_changes_cursor_value(), version=GOOGLE_DRIVE_CURSOR_VERSION),
        KnowledgeCursor(
            value=_valid_changes_cursor_value()[:20] + "!!!!" + _valid_changes_cursor_value()[24:],
            version=GOOGLE_DRIVE_CURSOR_VERSION,
        ),
        KnowledgeCursor(value=_valid_changes_cursor_value() + "!!!!", version=GOOGLE_DRIVE_CURSOR_VERSION),
        KnowledgeCursor(
            value=_valid_changes_cursor_value() + "=",
            version=GOOGLE_DRIVE_CURSOR_VERSION,
        ),
        KnowledgeCursor(
            value="A" * 24_577,
            version=GOOGLE_DRIVE_CURSOR_VERSION,
        ),
        KnowledgeCursor.model_construct(version=GOOGLE_DRIVE_CURSOR_VERSION),
        KnowledgeCursor.model_construct(value=_valid_changes_cursor_value()),
        _KnowledgeCursorSubclass(value=_valid_changes_cursor_value(), version=GOOGLE_DRIVE_CURSOR_VERSION),
    ],
)
async def test_cursor_malleability_rejected_before_provider_calls(
    cursor: KnowledgeCursor,
) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_noncanonical_cursor_json_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    payload = {
        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
        "scope_kind": "user",
        "drive_id": None,
        "phase": "changes",
        "inventory_page_token": None,
        "change_page_token": _SECRET_CHANGE,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(", ", ": ")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    cursor = KnowledgeCursor(value=encoded, version=GOOGLE_DRIVE_CURSOR_VERSION)
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_invalid_start_token_missing_value() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    bad_token = object.__new__(GoogleWorkspacePageToken)
    fake = _FakeGoogleWorkspaceIntegration(start_token=bad_token)  # type: ignore[arg-type]
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)
    assert len(fake.start_token_calls) == 1
    assert fake.inventory_calls == []


async def test_invalid_start_token_subclass() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    bad_token = _GoogleWorkspacePageTokenSubclass(value=_SECRET_START)
    fake = _FakeGoogleWorkspaceIntegration(start_token=bad_token)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)
    assert len(fake.start_token_calls) == 1
    assert fake.inventory_calls == []


async def test_invalid_start_token_non_str_value() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    bad_token = object.__new__(GoogleWorkspacePageToken)
    object.__setattr__(bad_token, "value", 123)
    fake = _FakeGoogleWorkspaceIntegration(start_token=bad_token)  # type: ignore[arg-type]
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)
    assert len(fake.start_token_calls) == 1
    assert fake.inventory_calls == []


async def test_inventory_page_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = _GoogleDriveItemPageSubclass(items=(_drive_item(),), next_page_token=None)
    fake = _FakeGoogleWorkspaceIntegration(inventory_pages=[page])  # type: ignore[list-item]
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_inventory_page_integer_next_token_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = GoogleDriveItemPage.model_construct(
        items=(_drive_item(),),
        next_page_token=123,
    )
    fake = _FakeGoogleWorkspaceIntegration(inventory_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_inventory_page_string_next_token_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = GoogleDriveItemPage.model_construct(
        items=(_drive_item(),),
        next_page_token="token",
    )
    fake = _FakeGoogleWorkspaceIntegration(inventory_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_inventory_page_nested_item_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _ItemSubclass(GoogleDriveItem):
        pass

    page = GoogleDriveItemPage.model_construct(
        items=(_ItemSubclass.model_construct(remote_id="x", scope=_USER_SCOPE),),
        next_page_token=None,
    )
    fake = _FakeGoogleWorkspaceIntegration(inventory_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_inventory_page_malformed_nested_item_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = GoogleDriveItemPage.model_construct(
        items=(GoogleDriveItem.model_construct(remote_id="x"),),
        next_page_token=None,
    )
    fake = _FakeGoogleWorkspaceIntegration(inventory_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_change_page_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = _GoogleDriveChangePageSubclass(
        changes=(),
        next_page_token=GoogleWorkspacePageToken(value=_SECRET_CHANGE),
        new_start_page_token=None,
    )
    fake = _FakeGoogleWorkspaceIntegration(change_pages=[page])  # type: ignore[list-item]
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_change_page_integer_next_token_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = GoogleDriveChangePage.model_construct(
        changes=(),
        next_page_token=123,
        new_start_page_token=None,
    )
    fake = _FakeGoogleWorkspaceIntegration(change_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_change_page_integer_new_start_token_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    page = GoogleDriveChangePage.model_construct(
        changes=(),
        next_page_token=None,
        new_start_page_token=456,
    )
    fake = _FakeGoogleWorkspaceIntegration(change_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_change_page_nested_item_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _ItemSubclass(GoogleDriveItem):
        pass

    item = _ItemSubclass.model_construct(remote_id="changed-1", scope=_USER_SCOPE)
    page = GoogleDriveChangePage.model_construct(
        changes=(
            GoogleDriveChange.model_construct(
                file_id="changed-1",
                scope=_USER_SCOPE,
                removed=False,
                changed_at=_MODIFIED,
                item=item,
            ),
        ),
        next_page_token=None,
        new_start_page_token=GoogleWorkspacePageToken(value=_SECRET_NEW_START),
    )
    fake = _FakeGoogleWorkspaceIntegration(change_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_change_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()

    class _ChangeSubclass(GoogleDriveChange):
        pass

    page = GoogleDriveChangePage.model_construct(
        changes=(
            _ChangeSubclass.model_construct(
                file_id="changed-1",
                scope=_USER_SCOPE,
                removed=False,
                changed_at=_MODIFIED,
                item=_drive_item(remote_id="changed-1"),
            ),
        ),
        next_page_token=None,
        new_start_page_token=GoogleWorkspacePageToken(value=_SECRET_NEW_START),
    )
    fake = _FakeGoogleWorkspaceIntegration(change_pages=[page])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_changes_cursor(),
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_plain_object_source_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=_integration(fake), source=object())  # type: ignore[arg-type]
    _assert_invalid_scope_boundary(exc_info, fake=fake)


async def test_source_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(fake),
            source=_KnowledgeSourceRefSubclass.model_construct(
                tenant_id="tenant-1",
                provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=GOOGLE_DRIVE_SOURCE_KIND,
                scope=KnowledgeSourceScope.model_construct(
                    remote_scope_id=GOOGLE_DRIVE_USER_SCOPE_ID,
                    remote_scope_type=GOOGLE_DRIVE_USER_SCOPE_TYPE,
                    safe_display_name="My Drive",
                    parameters={},
                ),
            ),
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


async def test_source_missing_model_field_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(fake),
            source=KnowledgeSourceRef.model_construct(
                tenant_id="tenant-1",
                provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            ),
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


async def test_source_malformed_nested_scope_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(fake),
            source=KnowledgeSourceRef.model_construct(
                tenant_id="tenant-1",
                provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=GOOGLE_DRIVE_SOURCE_KIND,
                scope=KnowledgeSourceScope.model_construct(
                    remote_scope_id=GOOGLE_DRIVE_USER_SCOPE_ID,
                    remote_scope_type=GOOGLE_DRIVE_USER_SCOPE_TYPE,
                ),
            ),
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


async def test_source_hostile_display_name_rejected() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DRIVE_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id=GOOGLE_DRIVE_USER_SCOPE_ID,
            remote_scope_type=GOOGLE_DRIVE_USER_SCOPE_TYPE,
            safe_display_name=_HostileStr("My Drive"),
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=_integration(fake), source=source)
    _assert_invalid_scope_boundary(exc_info, fake=fake)
    assert "hostile" not in f"{exc_info.value!r} {exc_info.value.safe_message}"


async def test_descriptor_subclass_rejected() -> None:
    descriptor = _descriptor_for_item(_drive_item())
    bad = _KnowledgeItemDescriptorSubclass.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=descriptor.metadata,
    )
    await _fetch_content_invalid_descriptor(bad)


async def test_descriptor_missing_nested_field_rejected() -> None:
    descriptor = _descriptor_for_item(_drive_item())
    bad = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=descriptor.metadata,
    )
    await _fetch_content_invalid_descriptor(bad)


async def test_descriptor_hostile_created_at_rejected() -> None:
    descriptor = _descriptor_for_item(_drive_item())
    metadata = dict(descriptor.metadata or {})
    metadata["created_at"] = _HostileDatetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    bad = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=metadata,
    )
    exc_info = await _fetch_content_invalid_descriptor(bad)
    assert "hostile" not in f"{exc_info.value!r} {exc_info.value.safe_message}"


async def test_descriptor_parent_ids_int_rejected() -> None:
    descriptor = _descriptor_for_item(_drive_item())
    metadata = dict(descriptor.metadata or {})
    metadata["parent_ids"] = [1]
    bad = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=metadata,
    )
    await _fetch_content_invalid_descriptor(bad)


async def test_descriptor_parent_ids_hostile_str_rejected() -> None:
    descriptor = _descriptor_for_item(_drive_item())
    metadata = dict(descriptor.metadata or {})
    metadata["parent_ids"] = [_HostileStr("parent-1")]
    bad = KnowledgeItemDescriptor.model_construct(
        identity=descriptor.identity,
        revision=descriptor.revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=descriptor.provenance,
        metadata=metadata,
    )
    exc_info = await _fetch_content_invalid_descriptor(bad)
    assert "hostile" not in f"{exc_info.value!r} {exc_info.value.safe_message}"


@pytest.mark.parametrize(
    "field_name",
    [
        "logical_key",
        "etag",
        "content_hash",
        "acl_hash",
        "safe_locator",
    ],
)
async def test_descriptor_fixed_fields_rejected(field_name: str) -> None:
    descriptor = _descriptor_for_item(_drive_item())
    identity = descriptor.identity
    revision = descriptor.revision
    provenance = descriptor.provenance
    if field_name == "logical_key":
        identity = KnowledgeItemIdentity(
            remote_id=identity.remote_id,
            parent_remote_id=identity.parent_remote_id,
            logical_key="fixed",
        )
    elif field_name == "etag":
        revision = KnowledgeItemRevision(
            version=revision.version,
            etag="fixed",
            content_hash=revision.content_hash,
            acl_hash=revision.acl_hash,
            updated_at=revision.updated_at,
        )
    elif field_name == "content_hash":
        revision = KnowledgeItemRevision(
            version=revision.version,
            etag=revision.etag,
            content_hash="fixed",
            acl_hash=revision.acl_hash,
            updated_at=revision.updated_at,
        )
    elif field_name == "acl_hash":
        revision = KnowledgeItemRevision(
            version=revision.version,
            etag=revision.etag,
            content_hash=revision.content_hash,
            acl_hash="fixed",
            updated_at=revision.updated_at,
        )
    else:
        provenance = KnowledgeItemProvenance(
            provider_id=provenance.provider_id,
            source_kind=provenance.source_kind,
            remote_id=provenance.remote_id,
            web_url=provenance.web_url,
            safe_locator="fixed",
        )
    bad = KnowledgeItemDescriptor(
        identity=identity,
        revision=revision,
        title=descriptor.title,
        item_type=descriptor.item_type,
        content_mode=descriptor.content_mode,
        content_available=descriptor.content_available,
        provenance=provenance,
        metadata=descriptor.metadata,
    )
    await _fetch_content_invalid_descriptor(bad)


async def test_fetch_permissions_malformed_descriptor_invalid_scope() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=KnowledgeItemDescriptor.model_construct(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.content_calls == []
    _assert_canonical_error_identity(exc_info.value)


async def test_wrong_integration_uses_canonical_identity() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=object(),
            source=_source(provider_id="jira", source_kind="issues"),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    _assert_canonical_error_identity(exc_info.value)


async def test_invalid_scope_uses_canonical_identity() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(fake),
            source=_source(provider_id="jira", source_kind="issues"),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    _assert_canonical_error_identity(exc_info.value)


@pytest.mark.parametrize(
    ("returned_item", "content_data"),
    [
        (
            _drive_item(
                kind=GoogleDriveItemKind.NATIVE_DOCUMENT,
                mime_type="application/vnd.google-apps.document",
                size_bytes=0,
                md5_checksum=None,
            ),
            b"exported-docx",
        ),
        (_drive_item(mime_type="application/octet-stream"), b"x"),
        (_drive_item(size_bytes=999), b"x"),
        (_drive_item(md5_checksum="other"), b"x"),
        (_drive_item(head_revision_id="other-rev"), b"x"),
        (_drive_item(can_download=False), b"x"),
    ],
)
async def test_fetch_content_revision_mismatch_rejected(
    returned_item: GoogleDriveItem,
    content_data: bytes,
) -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    requested = _drive_item()
    returned = returned_item.model_copy(
        update={
            "remote_id": requested.remote_id,
            "modified_at": requested.modified_at,
            "version": requested.version,
        }
    )
    profile = resolve_google_drive_content_profile(returned)
    bad_content = GoogleDriveFileContent.model_construct(
        item=returned,
        mode=profile.mode,
        content_mime_type=profile.content_mime_type,
        data=content_data,
        size_bytes=len(content_data),
        content_hash=hashlib.sha256(content_data).hexdigest(),
    )
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={
            requested.remote_id: bad_content,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_descriptor_for_item(requested),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    _assert_canonical_error_identity(exc_info.value)


async def test_fetch_content_tolerates_mutable_metadata() -> None:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    requested = _drive_item()
    returned = _drive_item(
        name="changed-name",
        parent_ids=("other-parent",),
        web_view_link="https://example.com/changed",
    )
    fake = _FakeGoogleWorkspaceIntegration(
        content_by_id={
            requested.remote_id: _file_content(item=returned, data=b"hello"),
        }
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_descriptor_for_item(requested),
    )
    assert content.binary == b"hello"
