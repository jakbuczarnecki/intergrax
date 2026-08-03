# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Google Workspace Drive knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import PrivateAttr

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
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
    GoogleDriveFileContent,
    resolve_google_drive_content_profile,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspacePageToken,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_DRIVE_CURSOR_VERSION,
    GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
    register_google_workspace_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import KnowledgeContentMode, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemStatus,
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_TENANT_ID = "tenant-1"
_BINDING_ID = "google-drive-binding"
_SHARED_DRIVE_ID = "shared-drive-proof"
_START_BEFORE = "start-before-inventory"
_INVENTORY_PAGE_2 = "inventory-page-2"
_START_AFTER = "start-after-incremental"

_CREATED = datetime(2026, 1, 10, 8, 0, 0, tzinfo=timezone.utc)
_MODIFIED_V1 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
_MODIFIED_V2 = datetime(2026, 2, 1, 14, 30, 0, tzinfo=timezone.utc)
_DOC_MODIFIED = datetime(2026, 1, 20, 9, 0, 0, tzinfo=timezone.utc)
_CHANGE_AT = datetime(2026, 2, 2, 8, 0, 0, tzinfo=timezone.utc)

_BLOB_V1_BYTES = b"%PDF-1.4 deterministic handbook v1 content"
_BLOB_V2_BYTES = b"%PDF-1.4 deterministic handbook v2 updated content"
_DOCX_BYTES = b"PK\x03\x04 deterministic docx export bytes"
_BLOB_MD5_V1 = "a1b2c3d4e5f6789012345678abcdef01"
_BLOB_MD5_V2 = "b2c3d4e5f6789012345678abcdef0123"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_SHARED_SCOPE = GoogleDriveScope(
    kind=GoogleDriveScopeKind.SHARED_DRIVE,
    drive_id=_SHARED_DRIVE_ID,
)


def _drive_item(
    *,
    remote_id: str,
    kind: GoogleDriveItemKind,
    name: str,
    mime_type: str,
    version: int,
    modified_at: datetime,
    size_bytes: int | None,
    md5_checksum: str | None,
    can_download: bool,
    head_revision_id: str | None = "rev-head",
) -> GoogleDriveItem:
    return GoogleDriveItem(
        remote_id=remote_id,
        scope=_SHARED_SCOPE,
        kind=kind,
        name=name,
        mime_type=mime_type,
        parent_ids=("root",),
        drive_id=_SHARED_DRIVE_ID,
        created_at=_CREATED,
        modified_at=modified_at,
        size_bytes=size_bytes,
        md5_checksum=md5_checksum,
        version=version,
        head_revision_id=head_revision_id,
        web_view_link=f"https://drive.google.com/file/d/{remote_id}/view",
        can_download=can_download,
    )


def _file_content(*, item: GoogleDriveItem, data: bytes) -> GoogleDriveFileContent:
    profile = resolve_google_drive_content_profile(item)
    return GoogleDriveFileContent(
        item=item,
        mode=profile.mode,
        content_mime_type=profile.content_mime_type,
        data=data,
        size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
    )


def _decode_drive_cursor_payload(cursor_value: str) -> dict[str, Any]:
    padding = "=" * (-len(cursor_value) % 4)
    raw = base64.urlsafe_b64decode(cursor_value + padding)
    return json.loads(raw.decode("utf-8"))


def _decode_persisted_checkpoint(cursor_value: str) -> dict[str, Any]:
    return _decode_drive_cursor_payload(cursor_value)


def _binding() -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=_BINDING_ID,
        tenant_id=_TENANT_ID,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DRIVE_SOURCE_KIND,
        connection_ref="conn-google-1",
        safe_display_name="Google Drive Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=_SHARED_DRIVE_ID,
            remote_scope_type=GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
            safe_display_name="Google Drive Proof",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )


class _GoogleDriveProviderScenario:
    """Deterministic provider scenario with strict call-order enforcement."""

    def __init__(self) -> None:
        self.event_log: list[str] = []
        self.start_token_calls: list[dict[str, Any]] = []
        self.inventory_calls: list[dict[str, Any]] = []
        self.change_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []

        self._blob_v1 = _drive_item(
            remote_id="blob-1",
            kind=GoogleDriveItemKind.BLOB,
            name="Handbook.pdf",
            mime_type="application/pdf",
            version=1,
            modified_at=_MODIFIED_V1,
            size_bytes=len(_BLOB_V1_BYTES),
            md5_checksum=_BLOB_MD5_V1,
            can_download=True,
            head_revision_id="rev-blob-1",
        )
        self._folder = _drive_item(
            remote_id="folder-1",
            kind=GoogleDriveItemKind.FOLDER,
            name="Policies",
            mime_type="application/vnd.google-apps.folder",
            version=1,
            modified_at=_MODIFIED_V1,
            size_bytes=None,
            md5_checksum=None,
            can_download=False,
            head_revision_id=None,
        )
        self._doc_v1 = _drive_item(
            remote_id="doc-1",
            kind=GoogleDriveItemKind.NATIVE_DOCUMENT,
            name="Operating Model",
            mime_type="application/vnd.google-apps.document",
            version=4,
            modified_at=_DOC_MODIFIED,
            size_bytes=0,
            md5_checksum=None,
            can_download=True,
            head_revision_id="rev-doc-4",
        )
        self._blob_v2 = _drive_item(
            remote_id="blob-1",
            kind=GoogleDriveItemKind.BLOB,
            name="Handbook.pdf",
            mime_type="application/pdf",
            version=2,
            modified_at=_MODIFIED_V2,
            size_bytes=len(_BLOB_V2_BYTES),
            md5_checksum=_BLOB_MD5_V2,
            can_download=True,
            head_revision_id="rev-blob-2",
        )

        self._inventory_page_1 = GoogleDriveItemPage(
            items=(self._blob_v1, self._folder),
            next_page_token=GoogleWorkspacePageToken(value=_INVENTORY_PAGE_2),
        )
        self._inventory_page_2 = GoogleDriveItemPage(
            items=(self._doc_v1,),
            next_page_token=None,
        )
        self._change_page = GoogleDriveChangePage(
            changes=(
                GoogleDriveChange(
                    file_id="blob-1",
                    scope=_SHARED_SCOPE,
                    removed=False,
                    changed_at=_CHANGE_AT,
                    item=self._blob_v2,
                ),
                GoogleDriveChange(
                    file_id="doc-1",
                    scope=_SHARED_SCOPE,
                    removed=True,
                    changed_at=_CHANGE_AT,
                    item=None,
                ),
            ),
            next_page_token=None,
            new_start_page_token=GoogleWorkspacePageToken(value=_START_AFTER),
        )
        self._content_by_key: dict[tuple[str, int], GoogleDriveFileContent] = {
            ("blob-1", 1): _file_content(item=self._blob_v1, data=_BLOB_V1_BYTES),
            ("doc-1", 4): _file_content(item=self._doc_v1, data=_DOCX_BYTES),
            ("blob-1", 2): _file_content(item=self._blob_v2, data=_BLOB_V2_BYTES),
        }

    def _assert_scope(self, scope: GoogleDriveScope) -> None:
        if scope.kind is not GoogleDriveScopeKind.SHARED_DRIVE:
            raise AssertionError(f"unexpected scope kind: {scope.kind}")
        if scope.drive_id != _SHARED_DRIVE_ID:
            raise AssertionError(f"unexpected drive_id: {scope.drive_id}")

    def _record(self, event: str) -> None:
        self.event_log.append(event)

    def read_drive_start_page_token(
        self,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleWorkspacePageToken:
        self._assert_scope(scope)
        if self.start_token_calls:
            raise AssertionError("start token must be requested exactly once")
        self._record("start_token")
        self.start_token_calls.append({"scope": scope})
        return GoogleWorkspacePageToken(value=_START_BEFORE)

    def read_drive_items_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken | None,
        limit: int,
    ) -> GoogleDriveItemPage:
        self._assert_scope(scope)
        self.inventory_calls.append(
            {"scope": scope, "page_token": page_token, "limit": limit}
        )
        if page_token is None:
            self._record("inventory_page_1")
            return self._inventory_page_1
        if page_token.value == _INVENTORY_PAGE_2:
            self._record("inventory_page_2")
            return self._inventory_page_2
        raise AssertionError(f"unexpected inventory page_token: {page_token.value!r}")

    def read_drive_changes_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken,
        limit: int,
    ) -> GoogleDriveChangePage:
        self._assert_scope(scope)
        self.change_calls.append(
            {"scope": scope, "page_token": page_token, "limit": limit}
        )
        if page_token.value != _START_BEFORE:
            raise AssertionError(f"unexpected changes page_token: {page_token.value!r}")
        self._record("changes_page")
        return self._change_page

    def read_drive_file_content(
        self,
        *,
        item: GoogleDriveItem,
        max_bytes: int,
    ) -> GoogleDriveFileContent:
        self.content_calls.append({"item": item, "max_bytes": max_bytes})
        key = (item.remote_id, item.version)
        if key not in self._content_by_key:
            raise AssertionError(f"unexpected content request for {key!r}")
        return self._content_by_key[key]


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
    _bound_fake: _GoogleDriveProviderScenario = PrivateAttr()

    @classmethod
    def from_scenario(
        cls, scenario: _GoogleDriveProviderScenario
    ) -> _BoundGoogleWorkspaceIntegration:
        bound = cls.from_client(_StubClientFamily(), enabled=True)
        bound._bound_fake = scenario
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
        return self._bound_fake.read_drive_file_content(item=item, max_bytes=max_bytes)


@dataclass
class _GoogleResolver:
    integration: GoogleWorkspaceCollaborationSuiteIntegration

    def resolve(self, *, source: object) -> GoogleWorkspaceCollaborationSuiteIntegration:
        return self.integration


def _build_runtime(
    *,
    scenario: _GoogleDriveProviderScenario,
    document_store: InMemoryDocumentStore,
    sink: IdempotentRecordingSink,
    binding: KnowledgeSourceBinding,
    owner_id: str,
) -> VendorKnowledgeSyncCoordinator:
    integration = _BoundGoogleWorkspaceIntegration.from_scenario(scenario)
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    registry = KnowledgeAdapterRegistry()
    register_google_workspace_drive_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id=_TENANT_ID,
        resolver=_GoogleResolver(integration=integration),
        adapter_registry=registry,
    )
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id=_TENANT_ID,
        owner_id=owner_id,
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
    )
    return coordinator


def _fresh_checkpoint_repo(
    document_store: InMemoryDocumentStore,
) -> DocumentStoreKnowledgeSyncCheckpointRepository:
    return DocumentStoreKnowledgeSyncCheckpointRepository(document_store)


def _fresh_state_repo(
    document_store: InMemoryDocumentStore,
) -> DocumentStoreKnowledgeRemoteItemStateRepository:
    return DocumentStoreKnowledgeRemoteItemStateRepository(document_store)


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_secrets(blob: str) -> None:
    forbidden = (
        _START_BEFORE,
        _INVENTORY_PAGE_2,
        _START_AFTER,
        "Authorization",
        "access_token",
        "refresh_token",
        "client_secret",
        "bearer",
        "x-goog-api-key",
    )
    for item in forbidden:
        assert item not in blob
    assert "credential_ref" not in blob


def _assert_permissions_and_metadata_safe(envelope_blob: str) -> None:
    parsed = json.loads(envelope_blob)
    assert parsed.get("permissions") is None
    descriptor = parsed.get("descriptor")
    if descriptor is not None:
        metadata = descriptor.get("metadata") or {}
        for key in metadata:
            assert "token" not in key.lower()
            assert "page" not in key.lower() or key in {"schema_version"}


def _envelope_for(batch, remote_id: str):
    return next(
        envelope for envelope in batch.envelopes if envelope.remote_id == remote_id
    )


@pytest.mark.asyncio
async def test_google_drive_facade_coordinator_restart_inventory_incremental_and_content() -> None:
    scenario = _GoogleDriveProviderScenario()
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = _binding()

    run_results: list[KnowledgeSyncRunResult] = []

    # Runtime A — first inventory page
    runtime_a = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-a",
    )
    first = await runtime_a.reconcile_once(binding_id=_BINDING_ID, restart=True)
    run_results.append(first)
    del runtime_a

    assert first.status is KnowledgeSyncRunStatus.COMPLETED
    assert first.mode is KnowledgeSyncMode.RECONCILIATION
    assert first.has_more is True
    assert first.checkpoint_advanced is True
    assert first.changes_count == 2
    assert first.active_count == 2
    assert first.tombstone_count == 0

    assert scenario.event_log == ["start_token", "inventory_page_1"]
    assert len(scenario.start_token_calls) == 1
    assert scenario.inventory_calls[0]["page_token"] is None

    batch_1 = sink.calls[0]
    blob_env = _envelope_for(batch_1, "blob-1")
    folder_env = _envelope_for(batch_1, "folder-1")
    assert blob_env.content is not None
    assert blob_env.content.mode is KnowledgeContentMode.BINARY
    assert blob_env.content.binary == _BLOB_V1_BYTES
    assert blob_env.permissions is None
    assert folder_env.content is None
    assert folder_env.permissions is None

    checkpoint_repo_a = _fresh_checkpoint_repo(document_store)
    checkpoint_after_page_1 = checkpoint_repo_a.get(
        tenant_id=_TENANT_ID, binding_id=_BINDING_ID
    )
    assert checkpoint_after_page_1 is not None
    assert checkpoint_after_page_1.binding_configuration_version == 1
    assert checkpoint_after_page_1.cursor is not None
    assert checkpoint_after_page_1.cursor.version == GOOGLE_DRIVE_CURSOR_VERSION
    decoded_page_1 = _decode_persisted_checkpoint(checkpoint_after_page_1.cursor.value)
    assert decoded_page_1 == {
        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
        "scope_kind": "shared_drive",
        "drive_id": _SHARED_DRIVE_ID,
        "phase": "inventory",
        "inventory_page_token": _INVENTORY_PAGE_2,
        "change_page_token": _START_BEFORE,
    }

    state_repo_a = _fresh_state_repo(document_store)
    for remote_id in ("blob-1", "folder-1"):
        state = state_repo_a.get(
            tenant_id=_TENANT_ID, binding_id=_BINDING_ID, remote_id=remote_id
        )
        assert state is not None
        assert state.status is KnowledgeRemoteItemStatus.ACTIVE

    # Runtime B — inventory continuation
    runtime_b = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-b",
    )
    second = await runtime_b.reconcile_once(binding_id=_BINDING_ID, restart=False)
    run_results.append(second)
    del runtime_b

    assert second.status is KnowledgeSyncRunStatus.COMPLETED
    assert second.mode is KnowledgeSyncMode.RECONCILIATION
    assert second.has_more is False
    assert second.checkpoint_advanced is True
    assert second.changes_count == 1
    assert len(scenario.start_token_calls) == 1
    assert scenario.inventory_calls[1]["page_token"].value == _INVENTORY_PAGE_2

    batch_2 = sink.calls[1]
    doc_env = _envelope_for(batch_2, "doc-1")
    assert doc_env.content is not None
    assert doc_env.content.mode is KnowledgeContentMode.BINARY
    assert doc_env.content.mime_type == _DOCX_MIME
    assert doc_env.content.binary == _DOCX_BYTES
    assert doc_env.content.content_hash == hashlib.sha256(_DOCX_BYTES).hexdigest()
    assert doc_env.permissions is None

    checkpoint_repo_b = _fresh_checkpoint_repo(document_store)
    checkpoint_after_page_2 = checkpoint_repo_b.get(
        tenant_id=_TENANT_ID, binding_id=_BINDING_ID
    )
    assert checkpoint_after_page_2 is not None
    decoded_page_2 = _decode_persisted_checkpoint(checkpoint_after_page_2.cursor.value)
    assert decoded_page_2 == {
        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
        "scope_kind": "shared_drive",
        "drive_id": _SHARED_DRIVE_ID,
        "phase": "changes",
        "inventory_page_token": None,
        "change_page_token": _START_BEFORE,
    }

    state_repo_b = _fresh_state_repo(document_store)
    doc_state = state_repo_b.get(
        tenant_id=_TENANT_ID, binding_id=_BINDING_ID, remote_id="doc-1"
    )
    assert doc_state is not None
    assert doc_state.status is KnowledgeRemoteItemStatus.ACTIVE

    delivery_ids_after_b = set(sink.durable_delivery_ids)
    assert len(delivery_ids_after_b) == 2
    assert len(delivery_ids_after_b) == len(sink.durable_delivery_ids)

    # Runtime C — incremental sync
    runtime_c = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-c",
    )
    third = await runtime_c.sync_once(binding_id=_BINDING_ID)
    run_results.append(third)
    del runtime_c

    assert third.status is KnowledgeSyncRunStatus.COMPLETED
    assert third.mode is KnowledgeSyncMode.INCREMENTAL
    assert third.has_more is False
    assert third.checkpoint_advanced is True
    assert third.changes_count == 2
    assert third.active_count == 1
    assert third.tombstone_count == 1
    assert scenario.change_calls[0]["page_token"].value == _START_BEFORE
    assert scenario.event_log == [
        "start_token",
        "inventory_page_1",
        "inventory_page_2",
        "changes_page",
    ]

    batch_3 = sink.calls[2]
    assert batch_3.envelopes[0].remote_id == "blob-1"
    assert batch_3.envelopes[0].change_kind.value == "upsert"
    assert batch_3.envelopes[1].remote_id == "doc-1"
    assert batch_3.envelopes[1].change_kind.value == "deleted"
    updated_blob = batch_3.envelopes[0]
    deleted_doc = batch_3.envelopes[1]
    assert updated_blob.descriptor is not None
    assert updated_blob.descriptor.revision.version == "2"
    assert updated_blob.content is not None
    assert updated_blob.content.binary == _BLOB_V2_BYTES
    assert updated_blob.content.content_hash == hashlib.sha256(_BLOB_V2_BYTES).hexdigest()
    assert updated_blob.permissions is None
    assert deleted_doc.descriptor is None
    assert deleted_doc.content is None
    assert deleted_doc.permissions is None

    content_remote_ids = [call["item"].remote_id for call in scenario.content_calls]
    assert content_remote_ids == ["blob-1", "doc-1", "blob-1"]
    content_versions = [call["item"].version for call in scenario.content_calls]
    assert content_versions == [1, 4, 2]
    assert "folder-1" not in content_remote_ids

    state_repo_c = _fresh_state_repo(document_store)
    blob_final = state_repo_c.get(
        tenant_id=_TENANT_ID, binding_id=_BINDING_ID, remote_id="blob-1"
    )
    folder_final = state_repo_c.get(
        tenant_id=_TENANT_ID, binding_id=_BINDING_ID, remote_id="folder-1"
    )
    doc_final = state_repo_c.get(
        tenant_id=_TENANT_ID, binding_id=_BINDING_ID, remote_id="doc-1"
    )
    assert blob_final is not None
    assert blob_final.status is KnowledgeRemoteItemStatus.ACTIVE
    assert blob_final.revision is not None
    assert blob_final.revision.version == "2"
    assert folder_final is not None
    assert folder_final.status is KnowledgeRemoteItemStatus.ACTIVE
    assert doc_final is not None
    assert doc_final.status is KnowledgeRemoteItemStatus.DELETED

    checkpoint_repo_c = _fresh_checkpoint_repo(document_store)
    checkpoint_final = checkpoint_repo_c.get(tenant_id=_TENANT_ID, binding_id=_BINDING_ID)
    assert checkpoint_final is not None
    decoded_final = _decode_persisted_checkpoint(checkpoint_final.cursor.value)
    assert decoded_final == {
        "schema_version": GOOGLE_DRIVE_CURSOR_VERSION,
        "scope_kind": "shared_drive",
        "drive_id": _SHARED_DRIVE_ID,
        "phase": "changes",
        "inventory_page_token": None,
        "change_page_token": _START_AFTER,
    }

    delivery_ids_final = set(sink.durable_delivery_ids)
    assert len(delivery_ids_final) == 3
    assert len(delivery_ids_final) == len({*sink.durable_delivery_ids})

    checkpoint_values = {
        checkpoint_after_page_1.cursor.value,
        checkpoint_after_page_2.cursor.value,
        checkpoint_final.cursor.value,
    }
    assert len(checkpoint_values) == 3

    public_proof = _public_blob(
        {
            "runs": [result.model_dump(mode="json") for result in run_results],
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
            "checkpoint_page_1": checkpoint_after_page_1.model_dump(mode="json"),
            "checkpoint_page_2": checkpoint_after_page_2.model_dump(mode="json"),
            "checkpoint_final": checkpoint_final.model_dump(mode="json"),
            "states": {
                "blob-1": blob_final.model_dump(mode="json"),
                "folder-1": folder_final.model_dump(mode="json"),
                "doc-1": doc_final.model_dump(mode="json"),
            },
        }
    )
    _assert_no_secrets(public_proof)
    for result in run_results:
        assert result.status is KnowledgeSyncRunStatus.COMPLETED
    for batch in sink.calls:
        for envelope in batch.envelopes:
            env_blob = json.dumps(envelope.model_dump(mode="json"))
            _assert_permissions_and_metadata_safe(env_blob)
