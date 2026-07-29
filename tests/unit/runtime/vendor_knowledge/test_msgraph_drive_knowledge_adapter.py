# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MsGraphDriveKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MSGRAPH_DRIVE_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    ConfluencePagesKnowledgeAdapter,
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MSGRAPH_DRIVE_CURSOR_VERSION,
    MSGRAPH_DRIVE_SCOPE_TYPE,
    MsGraphDriveKnowledgeAdapter,
    register_msgraph_drive_knowledge_adapter,
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

_DRIVE_ID = "b!drive-id-with-special-chars"
_OTHER_DRIVE_ID = "b!other-drive"
_QUOTED_DRIVE_ID = quote(_DRIVE_ID, safe="")
_SECRET_SKIP = "super-secret-skip-token"
_SECRET_DELTA = "super-secret-delta-token"
_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
    f"$skiptoken={_SECRET_SKIP}"
)
_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
    f"$deltatoken={_SECRET_DELTA}"
)
_OTHER_DRIVE_URL = (
    f"https://graph.microsoft.com/v1.0/drives/{quote(_OTHER_DRIVE_ID, safe='')}/root/delta?"
    f"$skiptoken=other"
)
_TS = datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc)


def _source(
    *,
    remote_scope_id: str = _DRIVE_ID,
    remote_scope_type: str = MSGRAPH_DRIVE_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = MSGRAPH_DRIVE_SOURCE_KIND,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name="Finance Drive",
            parameters=parameters or {},
        ),
    )


def _drive_item(
    *,
    remote_id: str = "file-1",
    drive_id: str = _DRIVE_ID,
    kind: MsGraphDriveItemKind = MsGraphDriveItemKind.FILE,
    name: str = "report.pdf",
    parent_remote_id: str | None = "parent-1",
    c_tag: str | None = '"ctag-1"',
    e_tag: str | None = '"etag-1"',
    mime_type: str | None = "application/pdf",
    size_bytes: int | None = 12,
    is_root: bool = False,
) -> MsGraphDriveItem:
    return MsGraphDriveItem(
        remote_id=remote_id,
        drive_id=drive_id,
        parent_remote_id=parent_remote_id,
        kind=kind,
        name=name,
        e_tag=e_tag,
        c_tag=c_tag,
        size_bytes=size_bytes,
        mime_type=mime_type,
        created_at=_TS,
        last_modified_at=_TS,
        web_url="https://contoso.sharepoint.com/file",
        is_root=is_root,
        deleted_state=None,
    )


def _delta_page(
    *,
    items: tuple[MsGraphDriveItem, ...],
    continuation_kind: MsGraphKnowledgeContinuationKind,
    url: str,
) -> MsGraphDriveDeltaPage:
    return MsGraphDriveDeltaPage(
        items=items,
        continuation=MsGraphKnowledgeContinuation(kind=continuation_kind, url=url),
    )


def _file_content(
    *,
    remote_id: str = "file-1",
    data: bytes = b"hello",
    c_tag: str = '"ctag-1"',
) -> MsGraphDriveFileContent:
    return MsGraphDriveFileContent(
        drive_id=_DRIVE_ID,
        remote_id=remote_id,
        content_revision=c_tag,
        data=data,
        size_bytes=len(data),
        mime_type="application/pdf",
        content_hash=hashlib.sha256(data).hexdigest(),
    )


class _FakeDriveCollaborationSuite(CollaborationSuite):
    def __init__(
        self,
        *,
        pages: list[MsGraphDriveDeltaPage] | None = None,
        content_by_id: dict[str, MsGraphDriveFileContent] | None = None,
        incremental_page: MsGraphDriveDeltaPage | None = None,
    ) -> None:
        self._pages = list(pages or [])
        self._content_by_id = dict(content_by_id or {})
        self._incremental_page = incremental_page
        self.delta_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.permission_calls = 0

    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphDriveDeltaPage:
        self.delta_calls.append(
            {
                "drive_id": drive_id,
                "continuation": continuation,
                "limit": limit,
            }
        )
        if (
            continuation is not None
            and continuation.kind == MsGraphKnowledgeContinuationKind.DELTA
            and self._incremental_page is not None
        ):
            return self._incremental_page
        if not self._pages:
            raise IntegrationDependencyError("no pages configured")
        return self._pages.pop(0)

    def read_drive_file_content(self, *, item: MsGraphDriveItem, max_bytes: int):
        self.content_calls.append({"item": item, "max_bytes": max_bytes})
        return self._content_by_id[item.remote_id]

    def read_drive_permissions_page(self, *, item, continuation=None):
        self.permission_calls += 1
        raise AssertionError("permissions must not be called")

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


def _integration(fake: _FakeDriveCollaborationSuite) -> Ms365GraphCollaborationSuiteIntegration:
    return Ms365GraphCollaborationSuiteIntegration.from_client(fake, enabled=True)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return KnowledgeCursor(value=encoded, version=MSGRAPH_DRIVE_CURSOR_VERSION)


def _file_descriptor(
    *,
    remote_id: str = "file-1",
    drive_id: str = _DRIVE_ID,
    c_tag: str = '"ctag-1"',
    metadata_drive_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    metadata_only: bool = False,
) -> KnowledgeItemDescriptor:
    base_metadata = {
        "drive_id": metadata_drive_id or drive_id,
        "drive_item_kind": "file",
        "size_bytes": 12,
        "mime_type": "application/pdf",
        "is_root": False,
        "created_at": _TS.isoformat(),
        "last_modified_at": _TS.isoformat(),
    }
    if metadata is not None:
        resolved_metadata = metadata if metadata_only else {**base_metadata, **metadata}
    else:
        resolved_metadata = base_metadata
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id, parent_remote_id="parent-1"),
        revision=KnowledgeItemRevision(version=c_tag, etag='"etag-1"', updated_at=_TS),
        title="report.pdf",
        item_type="msgraph_drive_file",
        content_mode=KnowledgeContentMode.BINARY,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
            remote_id=remote_id,
            web_url="https://contoso.sharepoint.com/file",
        ),
        metadata=resolved_metadata,
    )


def _assert_invalid_descriptor_boundary(exc_info: pytest.ExceptionInfo[VendorKnowledgeError]) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    assert err.__cause__ is None
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _DRIVE_ID,
        "file-1",
        "report.pdf",
        "https://contoso.sharepoint.com/file",
        "2026-05-29",
        "graph.microsoft.com",
        "Authorization",
    ):
        assert secret not in rendered


async def _fetch_content_invalid_descriptor(
    item: KnowledgeItemDescriptor,
) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    assert fake.content_calls == []
    _assert_invalid_descriptor_boundary(exc_info)
    return exc_info


async def test_adapter_identity() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    assert adapter.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == MSGRAPH_DRIVE_SOURCE_KIND


async def test_capabilities_exact_set() -> None:
    caps = MsGraphDriveKnowledgeAdapter().capabilities
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


async def test_valid_drive_scope_inspect() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite()
    info = await adapter.inspect_scope(integration=_integration(fake), source=_source())
    assert info.safe_display_name == "Finance Drive"
    assert fake.delta_calls == []


@pytest.mark.parametrize(
    ("source",),
    [
        (_source(provider_id="jira"),),
        (_source(integration_kind=IntegrationCategory.ISSUE_TRACKER),),
        (_source(source_kind="mail"),),
        (_source(remote_scope_type="sharepoint"),),
        (_source(parameters={"site": "x"}),),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=_integration(_FakeDriveCollaborationSuite()), source=source)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_malformed_drive_id_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeDriveCollaborationSuite()),
            source=_source(remote_scope_id="\x00bad"),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_empty_drive_id_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="",
            remote_scope_type=MSGRAPH_DRIVE_SCOPE_TYPE,
            safe_display_name="Finance Drive",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=_integration(_FakeDriveCollaborationSuite()), source=source)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_whitespace_drive_id_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="   ",
            remote_scope_type=MSGRAPH_DRIVE_SCOPE_TYPE,
            safe_display_name="Finance Drive",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=_integration(_FakeDriveCollaborationSuite()), source=source)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_wrong_integration_object_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=object(), source=_source())
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.parametrize("limit", [1, 200, 201, 1000])
async def test_limit_accepted_and_clamped(limit: int) -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=limit,
    )
    assert fake.delta_calls[0]["limit"] == min(limit, 200)


@pytest.mark.parametrize("limit", [0, 1001, True, False, "10", 10.5, None])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeDriveCollaborationSuite()),
            source=_source(),
            cursor=None,
            limit=limit,  # type: ignore[arg-type]
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


async def test_file_descriptor_mapping() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(remote_id="f-1", parent_remote_id="p-1"),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.remote_id == "f-1"
    assert descriptor.identity.parent_remote_id == "p-1"
    assert descriptor.title == "report.pdf"
    assert descriptor.item_type == "msgraph_drive_file"
    assert descriptor.content_mode is KnowledgeContentMode.BINARY
    assert descriptor.content_available is True
    assert descriptor.revision.version == '"ctag-1"'
    assert descriptor.revision.etag == '"etag-1"'
    assert descriptor.revision.updated_at == _TS
    assert descriptor.metadata is not None
    assert descriptor.metadata["drive_id"] == _DRIVE_ID
    assert descriptor.metadata["drive_item_kind"] == "file"
    assert descriptor.metadata["size_bytes"] == 12
    assert descriptor.metadata["mime_type"] == "application/pdf"
    assert descriptor.provenance.web_url == "https://contoso.sharepoint.com/file"


@pytest.mark.parametrize(
    ("kind", "item_type", "content_mode", "content_available"),
    [
        (MsGraphDriveItemKind.FOLDER, "msgraph_drive_folder", KnowledgeContentMode.STRUCTURED_RECORD, False),
        (MsGraphDriveItemKind.PACKAGE, "msgraph_drive_package", KnowledgeContentMode.STRUCTURED_RECORD, False),
        (MsGraphDriveItemKind.OTHER, "msgraph_drive_other", KnowledgeContentMode.STRUCTURED_RECORD, False),
    ],
)
async def test_non_file_descriptor_mapping(
    kind: MsGraphDriveItemKind,
    item_type: str,
    content_mode: KnowledgeContentMode,
    content_available: bool,
) -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(
                    _drive_item(
                        remote_id="node-1",
                        kind=kind,
                        name="Node",
                        c_tag=None,
                        mime_type=None,
                        size_bytes=None,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == item_type
    assert descriptor.content_mode is content_mode
    assert descriptor.content_available is content_available


async def test_file_without_ctag_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(c_tag=None),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
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
    assert exc_info.value.safe_message == "Microsoft Graph Drive file revision is missing"


async def test_tombstone_mapping() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    tombstone = MsGraphDriveItem(
        remote_id="gone-1",
        drive_id=_DRIVE_ID,
        kind=MsGraphDriveItemKind.DELETED,
    )
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(tombstone,),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.DELETED
    assert change.descriptor is None


async def test_initial_read_next_page_cursor() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
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
    assert _SECRET_SKIP not in page.next_cursor.value
    assert "skiptoken" not in page.next_cursor.value


async def test_delta_checkpoint_semantics() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
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
    assert _SECRET_DELTA not in page.proposed_checkpoint.value


async def test_next_incremental_from_delta_cursor() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(remote_id="changed-1"),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_DRIVE_CURSOR_VERSION,
            "drive_id": _DRIVE_ID,
            "continuation_kind": "delta",
            "continuation_url": _DELTA_URL,
        }
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    continuation = fake.delta_calls[0]["continuation"]
    assert continuation is not None
    assert continuation.kind == MsGraphKnowledgeContinuationKind.DELTA
    assert continuation.url == _DELTA_URL


async def test_cursor_url_hidden_from_repr() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.next_cursor is not None
    rendered = repr(page.next_cursor)
    assert _SECRET_SKIP not in rendered
    assert _NEXT_URL not in rendered


@pytest.mark.parametrize(
    ("cursor",),
    [
        (KnowledgeCursor(value="not-base64", version=MSGRAPH_DRIVE_CURSOR_VERSION),),
        (KnowledgeCursor(value="!!!", version=MSGRAPH_DRIVE_CURSOR_VERSION),),
        (KnowledgeCursor(value=_encode_cursor({"bad": 1}).value, version="other.v1"),),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": "msgraph.drive.cursor.v1",
                        "drive_id": _DRIVE_ID,
                        "continuation_kind": "next_page",
                        "continuation_url": "",
                    }
                ).value,
                version=MSGRAPH_DRIVE_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": "msgraph.drive.cursor.v1",
                        "drive_id": _OTHER_DRIVE_ID,
                        "continuation_kind": "delta",
                        "continuation_url": _OTHER_DRIVE_URL,
                    }
                ).value,
                version=MSGRAPH_DRIVE_CURSOR_VERSION,
            ),
        ),
    ],
)
async def test_invalid_cursor_rejected(cursor: KnowledgeCursor) -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeDriveCollaborationSuite()),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_cross_drive_item_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(drive_id=_OTHER_DRIVE_ID),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
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


async def test_duplicate_remote_id_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    item = _drive_item(remote_id="dup-1")
    fake = _FakeDriveCollaborationSuite(
        pages=[
            MsGraphDriveDeltaPage.model_construct(
                items=(item, item),
                continuation=MsGraphKnowledgeContinuation(
                    kind=MsGraphKnowledgeContinuationKind.DELTA,
                    url=_DELTA_URL,
                ),
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


async def test_fetch_content_valid_file() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    data = b"\x00\xffbinary"
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=data)}
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(),
    )
    assert content.mode is KnowledgeContentMode.BINARY
    assert content.binary == data
    assert content.mime_type == "application/pdf"
    assert content.content_hash == hashlib.sha256(data).hexdigest()
    assert fake.content_calls[0]["item"].remote_id == "file-1"
    assert fake.content_calls[0]["item"].drive_id == _DRIVE_ID
    assert fake.content_calls[0]["item"].c_tag == '"ctag-1"'


async def test_fetch_content_uses_descriptor_not_metadata_override() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    item = _file_descriptor(metadata_drive_id=_OTHER_DRIVE_ID)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_fetch_content_provider_mismatch_rejected() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={
            "file-1": MsGraphDriveFileContent(
                drive_id=_OTHER_DRIVE_ID,
                remote_id="file-1",
                content_revision='"ctag-1"',
                data=b"x",
                size_bytes=1,
                mime_type="application/pdf",
                content_hash=hashlib.sha256(b"x").hexdigest(),
            )
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_file_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.parametrize(
    "created_at",
    [
        123,
        [],
        {},
        "",
        "   ",
        "not-a-date",
        "2026-05-29T10:15:30",
    ],
)
async def test_fetch_content_rejects_malformed_created_at(created_at: object) -> None:
    await _fetch_content_invalid_descriptor(
        _file_descriptor(metadata={"created_at": created_at}),
    )


@pytest.mark.parametrize(
    "is_root",
    [
        0,
        1,
        "true",
        "false",
        [],
        {},
    ],
)
async def test_fetch_content_rejects_malformed_is_root(is_root: object) -> None:
    await _fetch_content_invalid_descriptor(
        _file_descriptor(metadata={"is_root": is_root}),
    )


async def test_fetch_content_accepts_created_at_with_timezone_offset() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    created_at = "2026-05-29T10:15:30+02:00"
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(metadata={"created_at": created_at}),
    )
    provider_item = fake.content_calls[0]["item"]
    assert provider_item.created_at is not None
    assert provider_item.created_at.tzinfo is not None
    assert provider_item.created_at.utcoffset() is not None


async def test_fetch_content_accepts_created_at_with_z_suffix() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    created_at = "2026-05-29T10:15:30Z"
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(metadata={"created_at": created_at}),
    )
    provider_item = fake.content_calls[0]["item"]
    assert provider_item.created_at is not None
    assert provider_item.created_at.tzinfo is not None
    assert provider_item.created_at.utcoffset() is not None


async def test_fetch_content_accepts_is_root_true() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(metadata={"is_root": True}),
    )
    assert fake.content_calls[0]["item"].is_root is True


async def test_fetch_content_absent_created_at_maps_to_none() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(
            metadata={
                "drive_id": _DRIVE_ID,
                "drive_item_kind": "file",
                "size_bytes": 12,
                "mime_type": "application/pdf",
                "is_root": False,
            },
            metadata_only=True,
        ),
    )
    assert fake.content_calls[0]["item"].created_at is None


async def test_fetch_content_absent_is_root_maps_to_false() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        content_by_id={"file-1": _file_content(remote_id="file-1", data=b"x")}
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(
            metadata={
                "drive_id": _DRIVE_ID,
                "drive_item_kind": "file",
                "size_bytes": 12,
                "mime_type": "application/pdf",
                "created_at": _TS.isoformat(),
            },
            metadata_only=True,
        ),
    )
    assert fake.content_calls[0]["item"].is_root is False


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("metadata", "not-a-dict"),
        ("revision", "not-a-revision"),
        ("identity", "not-an-identity"),
    ],
)
async def test_fetch_content_rejects_malformed_descriptor_shape(
    field_name: str,
    field_value: object,
) -> None:
    base = _file_descriptor()
    item = KnowledgeItemDescriptor.model_construct(
        identity=base.identity,
        revision=base.revision,
        title=base.title,
        item_type=base.item_type,
        content_mode=base.content_mode,
        content_available=base.content_available,
        provenance=base.provenance,
        metadata=base.metadata,
    )
    object.__setattr__(item, field_name, field_value)
    await _fetch_content_invalid_descriptor(item)


@pytest.mark.parametrize(
    ("item",),
    [
        (
            KnowledgeItemDescriptor(
                identity=KnowledgeItemIdentity(remote_id="folder-1"),
                revision=KnowledgeItemRevision(version="v", updated_at=_TS),
                title="Folder",
                item_type="msgraph_drive_folder",
                content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
                content_available=False,
                provenance=KnowledgeItemProvenance(
                    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                    source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
                    remote_id="folder-1",
                ),
                metadata={"drive_id": _DRIVE_ID, "drive_item_kind": "folder", "is_root": False},
            ),
        ),
    ],
)
async def test_fetch_content_rejects_non_files(item: KnowledgeItemDescriptor) -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_FakeDriveCollaborationSuite()),
            source=_source(),
            item=item,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_fetch_permissions_unsupported() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_file_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert fake.permission_calls == 0


async def test_registry_registration() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = register_msgraph_drive_knowledge_adapter(registry)
    assert isinstance(adapter, MsGraphDriveKnowledgeAdapter)
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
    register_msgraph_drive_knowledge_adapter(registry)
    with pytest.raises(ValueError):
        register_msgraph_drive_knowledge_adapter(registry)


async def test_integration_dependency_error_translated() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()

    class _FailingSuite(_FakeDriveCollaborationSuite):
        def read_drive_delta_page(self, *, drive_id, continuation=None, limit):
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
    adapter = MsGraphDriveKnowledgeAdapter()

    class _FailingSuite(_FakeDriveCollaborationSuite):
        def read_drive_delta_page(self, *, drive_id, continuation=None, limit):
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


async def test_public_objects_do_not_leak_secrets() -> None:
    adapter = MsGraphDriveKnowledgeAdapter()
    fake = _FakeDriveCollaborationSuite(
        pages=[
            _delta_page(
                items=(_drive_item(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ],
        content_by_id={"file-1": _file_content()},
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    blob = json.dumps(page.model_dump(mode="json"))
    for secret in (_SECRET_SKIP, _SECRET_DELTA, _NEXT_URL, "skiptoken", "deltatoken", "Authorization"):
        assert secret not in blob
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_file_descriptor(),
    )
    content_blob = json.dumps(content.model_dump(mode="json"))
    assert _NEXT_URL not in content_blob
    assert "hello" not in repr(page)
    err = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_CURSOR,
        safe_message="Microsoft Graph Drive knowledge cursor is invalid",
        provider_id=adapter.provider_id,
        source_kind=adapter.source_kind,
        retryable=False,
    )
    assert _SECRET_SKIP not in repr(err)
    assert _SECRET_SKIP not in str(err)
